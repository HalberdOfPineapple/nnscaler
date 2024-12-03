import os
import pdb
import torch
import triton
import triton.language as tl

from fwd_utils import (
    _triton_mixed_sparse_attention, pad_tensor, 
    profile, get_rep
)
from bwd_utils import _bwd_preprocess

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
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh

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
        n_mask = cols < context_size

        # -- load k, v --
        k_ptrs = K + (cols[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (cols[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.)

        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (cols[None, :])
            qk = tl.where(causal_mask & m_mask, float(0.), float("-inf")).to(tl.float32)
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_ptrs = DV + (cols[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        tl.atomic_add(dv_ptrs, dv_vals, mask=n_mask[:, None])

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - D_i[:, None]
        dp += tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_ptrs = DK + (cols[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        tl.atomic_add(dk_ptrs, dk_vals, mask=n_mask[:, None])

        # compute dq
        dq += tl.dot(ds.to(Q.dtype.element_ty), k)

    for start_n in range(0, num_cols, BLOCK_N):
        # the key difference is that cols, as the indices, are stored in and loaded from cols_ptr
        # At each iteration, a block-sized chunk of column **indices** are loaded from cols_ptr, which can be discontinuous
        # But we just load the indices block by block, equivalent to translating the non-sparse columns together
        n_mask = (start_n + offs_n < num_cols)
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)

        # -- load k, v --
        k_ptrs = K + (cols[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (cols[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.)

        # Computer qk
        qk = tl.where(m_mask & n_mask, float(0.), float("-inf"))
        qk = qk + tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_ptrs = DV + (cols[:, None] * stride_vn + offs_k[None] * stride_vk)
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        tl.atomic_add(dv_ptrs, dv_vals, mask=n_mask[:, None])

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - D_i[:, None]
        dp = dp + tl.dot(do, tl.trans(v))
        
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_ptrs = DK + (cols[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        tl.atomic_add(dk_ptrs, dk_vals, mask=n_mask[:, None])

        # compute dq
        dq = dq + tl.dot(ds.to(Q.dtype.element_ty), k)

    dq_ptrs = DQ + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    tl.store(dq_ptrs, dq, mask=m_mask)

class MFMB(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        *args,
        **kwargs,
    ):
        REP = get_rep()
        if REP > 0: 
            print('-' * 40)
            print(f"{__class__.__name__} FWD Profiling")
            profile(lambda: MFMB.forward_(ctx, *args, **kwargs), 'Triton MFRBRow FWD', warmup=25, rep=REP)
            print('-' * 40)
        return MFMB.forward_(ctx, *args, **kwargs)
    

    @staticmethod
    def backward(ctx, do):
        REP = get_rep()
        if REP > 0: 
            print(f"{__class__.__name__} BWD Profiling")
            profile(lambda: MFMB.backward_(ctx, do), 'Triton MFRBRow BWD', warmup=25, rep=REP)
            print('-' * 40)
        return MFMB.backward_(ctx, do)


    @staticmethod
    def forward_(
        ctx, 
        query, key, value, 
        block_count, block_offset, column_count, column_index,
        seqlens: torch.Tensor,
        causal: bool,
        block_size_M: int = 64,
        block_size_N: int = 64,
    ):
        _,  _, context_size, head_dim = query.shape
        ctx.context_size = context_size
        ctx.head_dim = head_dim

        # the input version of Q, K and V are padded to be divisible by BLOCK_SIZE_M
        # the output and softmax_lse are also padded
        sm_scale = head_dim ** -0.5

        query, key, value = (
            pad_tensor(query, context_size, head_dim, block_size_M), 
            pad_tensor(key, context_size, head_dim, block_size_M), 
            pad_tensor(value, context_size, head_dim, block_size_M)
        )

        o, softmax_lse = _triton_mixed_sparse_attention(
            query, key, value, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, context_size, head_dim,
            block_size_M, block_size_N,
        )

        ctx.save_for_backward(
            query, key, value, o, softmax_lse,
            block_count, block_offset, column_count, column_index,
        )
        
        ctx.block_size_M = block_size_M
        ctx.block_size_N = block_size_N
        ctx.causal = causal
        return o

    @staticmethod
    def backward_(ctx, do):
        (
            q, k, v, o, L,
            block_count, block_offset, column_count, column_index
        ) = ctx.saved_tensors
        block_size_M, block_size_N = ctx.block_size_M, ctx.block_size_N
        head_dim, context_size = ctx.head_dim, ctx.context_size
        causal = ctx.causal
        sm_scale = 1.0 / (head_dim ** 0.5)

        grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)

        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        delta = torch.zeros_like(L)

        torch.cuda.synchronize()
        print(f"{__class__.__name__} | Starting Preprocess Kernel...", end=" ") 
        _bwd_preprocess[(grid[0] * grid[1], )](
            o, do,
            delta,
            BLOCK_M=block_size_M, 
            D_HEAD=q.shape[-1],
        )
        print("Done.")
        torch.cuda.synchronize()


        torch.cuda.synchronize()
        print(f"{__class__.__name__} | Starting Triton Kernel...", end=" ")
        _mbwd_kernel[(grid[0], grid[1],)](
            q, k, v,
            sm_scale,
            context_size,
            block_count, block_offset, column_count, column_index,
            o, do,
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], 0,
            grid[0], triton.cdiv(k.shape[2], block_size_N), 
            block_offset.shape[-1], column_index.shape[-1],
            BLOCK_M=block_size_M, 
            BLOCK_N=block_size_N,
            BLOCK_DMODEL=q.shape[-1], 
            CAUSAL=causal,
            num_warps=8,
            num_stages=2,
        )
        print("Done.")
        torch.cuda.synchronize()


        return dq[..., :context_size, :head_dim], dk[..., :context_size, :head_dim], dv[..., :context_size, :head_dim], None, None, None, None, None, None



class MFMBTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        query, key, value, 
        block_count, block_offset, column_count, column_index,
        seqlens: torch.Tensor,
        causal: bool,
        block_size_M: int = 64,
        block_size_N: int = 64,
    ):
        _,  _, context_size, head_dim = query.shape
        ctx.context_size = context_size
        ctx.head_dim = head_dim

        # the input version of Q, K and V are padded to be divisible by BLOCK_SIZE_M
        # the output and softmax_lse are also padded
        sm_scale = head_dim ** -0.5

        query, key, value = (
            pad_tensor(query, context_size, head_dim, block_size_M), 
            pad_tensor(key, context_size, head_dim, block_size_M), 
            pad_tensor(value, context_size, head_dim, block_size_M)
        )

        o, softmax_lse = _triton_mixed_sparse_attention(
            query, key, value, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, context_size, head_dim,
            block_size_M, block_size_N,
        )

        ctx.save_for_backward(
            query, key, value, o, softmax_lse,
            block_count, block_offset, column_count, column_index,
        )
        
        ctx.block_size_M = block_size_M
        ctx.block_size_N = block_size_N
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        (
            query, key, value, o, L,
            block_count, block_offset, column_count, column_index
        ) = ctx.saved_tensors
        block_size_M, block_size_N = ctx.block_size_M, ctx.block_size_N
        
        head_dim, context_size = ctx.head_dim, ctx.context_size
        causal = ctx.causal
        sm_scale = 1.0 / (head_dim ** 0.5)
        qk_scale = sm_scale * 1.44269504  # ln(2)

        DO = do.contiguous()
        DQ = torch.zeros_like(query, dtype=torch.float32)
        DK = torch.zeros_like(key)
        DV = torch.zeros_like(value)
        D = torch.sum(o * do, dim=-1).to(query.device)


        P_SEQ = 0
        Z, H, N_CTX, D_MODEL = query.shape
        NUM_BLOCKS_Q = triton.cdiv(query.shape[2], block_size_M)
        for batch_idx in range(Z):
            for head_idx in range(H):
                q = query[batch_idx, head_idx, :, :]
                k = key[batch_idx, head_idx, :, :]
                v = value[batch_idx, head_idx, :, :]
                do = DO[batch_idx, head_idx, :, :]
                l = L[batch_idx, head_idx, :]
                d = D[batch_idx, head_idx, :]
                for block_m_idx in range(NUM_BLOCKS_Q):
                    block_cnt = block_count[batch_idx, head_idx, block_m_idx]
                    block_off = block_offset[batch_idx, head_idx, block_m_idx, :block_cnt]
                    col_cnt = column_count[batch_idx, head_idx, block_m_idx]
                    col_idx = column_index[batch_idx, head_idx, block_m_idx, :col_cnt]

                    # the rows will automatically stop at `context_size` without the need for masking out the padded rows
                    rows = torch.arange(
                        block_m_idx * block_size_M, min((block_m_idx + 1) * block_size_M, context_size), 
                        dtype=torch.int64
                    ).to(query.device)
                    q_block, do_block = q[rows, :], do[rows, :]
                    l_block = l[rows]
                    d_block = d[rows]

                    dq = torch.zeros([len(rows), D_MODEL]).to(DQ.dtype).to(query.device)

                    # print(f"Row Block {block_m_idx} | ", end="")
                    for block_start_idx in range(block_cnt):
                        # if block_start_idx == 0: print("Block Start Indices = [", end="")
                        start_n = block_off[block_start_idx]
                        # print(f"{start_n}, ", end="")

                        cols = torch.arange(start_n, min(start_n + block_size_N, context_size), dtype=torch.int64).to(query.device)
                        k_block, v_block = k[cols, :], v[cols, :]
                        if causal:
                            causal_mask = (P_SEQ + rows[:, None]) >= cols[None, :]
                            qk = torch.where(causal_mask, torch.tensor(0.), torch.tensor(float("-inf"))).to(query.device).to(torch.float32)
                        else:
                            qk = torch.zeros([len(rows), len(cols)]).to(query.device)
                        
                        qk += torch.matmul(q_block, k_block.transpose(-1, -2))
                        qk *= qk_scale
                        p = torch.exp2(qk - l_block[:, None])

                        dv_block = torch.matmul(p.transpose(-1, -2).to(query.dtype), do_block)
                        DV[batch_idx, head_idx, cols, :] += dv_block

                        dp = torch.zeros([len(rows), len(cols)]).to(query.device).to(torch.float32) - d_block[:, None]
                        dp += do_block @ v_block.transpose(-1, -2)

                        ds = p * dp * sm_scale
                        DK[batch_idx, head_idx, cols, :] += torch.matmul(ds.transpose(-1, -2).to(query.dtype), q_block)

                        dq += torch.matmul(ds.to(query.dtype), k_block)
                    # print("]")
                    

                    for start_n in range(0, col_cnt, block_size_N):
                        # cols will always load from within the range of (0, col_cnt) -> no need for n_mask to mask out the padded columns
                        # if start_n == 0: print("Sparse Block Start Indices = [", end="")

                        cols = col_idx[start_n:min(start_n + block_size_N, col_cnt)]
                        k_block, v_block = k[cols, :], v[cols, :]
                        print(f"rows = {rows}, cols = {cols}")
                        # print(f"{cols}, ", end="")

                        qk = torch.zeros([len(rows), len(cols)]).to(query.device)

                        qk += torch.matmul(q_block, k_block.transpose(-1, -2))
                        qk *= qk_scale
                        p = torch.exp2(qk - l_block[:, None])

                        dv_block = torch.matmul(p.transpose(-1, -2).to(query.dtype), do_block)
                        DV[batch_idx, head_idx, cols, :] += dv_block

                        dp = torch.zeros([len(rows), len(cols)]).to(query.device).to(torch.float32) - d_block[:, None]
                        dp += do_block @ v_block.transpose(-1, -2)

                        ds = p * dp * sm_scale
                        DK[batch_idx, head_idx, cols, :] += torch.matmul(ds.transpose(-1, -2).to(query.dtype), q_block)

                        dq += torch.matmul(ds.to(query.dtype), k_block)
                    # print("]")
                    DQ[batch_idx, head_idx, rows, :] = dq

        return DQ[..., :context_size, :head_dim], DK[..., :context_size, :head_dim], DV[..., :context_size, :head_dim], None, None, None, None, None, None



@triton.jit
def _mbwd_dt_kernel(
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

class MFMBDT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        query, key, value, 
        block_count, block_offset, column_count, column_index,
        seqlens: torch.Tensor,
        causal: bool,
        block_size_M: int = 64,
        block_size_N: int = 64,
    ):
        _,  _, context_size, head_dim = query.shape
        ctx.context_size = context_size
        ctx.head_dim = head_dim

        # the input version of Q, K and V are padded to be divisible by BLOCK_SIZE_M
        # the output and softmax_lse are also padded
        sm_scale = head_dim ** -0.5

        # Check column_index
        if torch.any(column_index < 0) or torch.any(column_index >= context_size):
            raise ValueError("Ilegal column index detected.")
        else:
            print("Column Index is legal.")

        query, key, value = (
            pad_tensor(query, context_size, head_dim, block_size_M), 
            pad_tensor(key, context_size, head_dim, block_size_M), 
            pad_tensor(value, context_size, head_dim, block_size_M)
        )

        o, softmax_lse = _triton_mixed_sparse_attention(
            query, key, value, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, context_size, head_dim,
            block_size_M, block_size_N,
        )

        ctx.save_for_backward(
            query, key, value, o, softmax_lse,
            block_count, block_offset, column_count, column_index,
        )
        
        ctx.block_size_M = block_size_M
        ctx.block_size_N = block_size_N
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        (
            q, k, v, o, L,
            block_count, block_offset, column_count, column_index
        ) = ctx.saved_tensors
        block_size_M, block_size_N = ctx.block_size_M, ctx.block_size_N
        head_dim, context_size = ctx.head_dim, ctx.context_size
        causal = ctx.causal
        sm_scale = 1.0 / (head_dim ** 0.5)

        grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)

        do = do.contiguous()
        dq, dk, dv = (
            torch.zeros_like(q).to(torch.float32), 
            torch.zeros_like(k).to(torch.float32), 
            torch.zeros_like(v).to(torch.float32)
        )
        delta = torch.zeros_like(L)

        torch.cuda.synchronize()
        print(f"{__class__.__name__} | Starting Preprocess Kernel...", end=" ") 
        _bwd_preprocess[(grid[0] * grid[1], )](
            o, do,
            delta,
            BLOCK_M=block_size_M, 
            D_HEAD=q.shape[-1],
        )
        print("Done.")
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        print(f"{__class__.__name__} | Starting Triton Kernel...", end=" ") 
        _mbwd_dt_kernel[(grid[0], grid[1],)](
            q, k, v,
            sm_scale, context_size,
            block_count, block_offset, column_count, column_index,
            o, do,
            dq, dk, dv,
            L, delta,
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
            CAUSAL=causal,
            num_warps=32,
            num_stages=2,
        )
        print("Done.")
        torch.cuda.synchronize()

        return (
            dq[:, :, :context_size, :head_dim].to(q.dtype), 
            dk[:, :, :context_size, :head_dim].to(k.dtype), 
            dv[:, :, :context_size, :head_dim].to(v.dtype), 
            None, None, None, None, None, None
        )

