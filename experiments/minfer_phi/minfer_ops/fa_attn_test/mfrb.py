import torch
import triton
import triton.language as tl

from fwd_utils import (
    _triton_mixed_sparse_attention, pad_tensor, 
    profile, get_rep
)
from bwd_utils import _bwd_preprocess

@triton.jit
def _bwd_kernel(
    Q, K, V, 
    sm_scale, context_size,
    Out, DO,
    DQ, DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX, P_SEQ,
    num_block_q, num_block_kv,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    off_hz = tl.program_id(0)
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
    for start_n in range(0, num_block_kv):
        if CAUSAL:
            lo = tl.math.max(start_n * BLOCK_M - P_SEQ, 0)
        else:
            lo = 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dk amd dv
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block_q * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                causal_mask = (P_SEQ + offs_m_curr[:, None]) >= (offs_n[None, :])
                m_mask = offs_m_curr[:, None] < context_size
                qk = tl.where(causal_mask & m_mask, float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = tl.math.exp2(qk - l_i[:, None])

            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)

            
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        tl.store(dk_ptrs, dk)
        tl.store(dv_ptrs, dv)

class MFRB(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        query, key, value, 
        block_count, block_offset, column_count, column_index,
        seqlens: torch.Tensor,
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
            query, key, value, o, softmax_lse
        )
        
        ctx.block_size_M = block_size_M
        ctx.block_size_N = block_size_N
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        block_size_M, block_size_N = ctx.block_size_M, ctx.block_size_N
        head_dim, context_size = ctx.head_dim, ctx.context_size
        sm_scale = 1.0 / (head_dim ** 0.5)

        print('-' * 20)
        grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)

        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)

        _bwd_preprocess[(grid[0] * grid[1], )](
            o, do,
            delta,
            BLOCK_M=block_size_M, 
            D_HEAD=q.shape[-1],
        )

        # D = torch.sum(do * o, dim=-1).to(L.dtype)
        # if not torch.allclose(delta, D, atol=ATOL, rtol=RTOL):
        #     print(f"Triton-calculated Delta is not close to D")
        # else:
        #     print(f"Delta correctly calculated")

        _bwd_kernel[(grid[1],)](
            q, k, v,
            sm_scale,
            context_size,
            o, do,
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], 0,
            grid[0], triton.cdiv(k.shape[2], block_size_N),
            BLOCK_M=block_size_M, 
            BLOCK_N=block_size_N,
            BLOCK_DMODEL=q.shape[-1], 
            CAUSAL=True,
            num_warps=8,
            num_stages=1,
        )


        return dq[..., :context_size, :head_dim], dk[..., :context_size, :head_dim], dv[..., :context_size, :head_dim], None, None, None, None, None





@triton.jit
def _bwd_row_kernel(
    Q, K, V, 
    sm_scale, context_size,
    Out, DO,
    DQ, DK, DV,
    L,
    D, # [BATCH, N_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk, 
    Z, H, N_CTX, P_SEQ,
    num_block_q, num_block_kv,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    off_hz = tl.program_id(0)
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
    for start_m in range(0, num_block_q):
        # initialize row/col offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
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
        
        # num_block_kv_curr = (start_m * BLOCK_M + BLOCK_N - 1) // BLOCK_N
        # for start_n in range(0, num_block_kv_curr):
        for start_n in range(0, num_block_kv):
            offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # k and v stay in SRAM throughout
            k_ptrs  = K  + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
            dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
            k = tl.load(k_ptrs)

            v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
            dv_ptrs = DV + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
            v = tl.load(v_ptrs)
            dv = tl.load(dv_ptrs)

            if CAUSAL:
                causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :])
                qk = tl.where(causal_mask & m_mask, float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            p = tl.math.exp2(qk - l_i[:, None])

            # compute dv
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            tl.store(dv_ptrs, dv)

            # compute dp = dot(v, do)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - D_i[:, None]
            dp += tl.dot(do, tl.trans(v))
            
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale

            # compute dk = dot(ds.T, q)
            dk = tl.load(dk_ptrs)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            tl.store(dk_ptrs, dk)

            # compute dq
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)

        dq_ptrs = DQ + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        tl.store(dq_ptrs, dq)

class MFRBRow(torch.autograd.Function):
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
            profile(lambda: MFRBRow.forward_(ctx, *args, **kwargs), 'Triton MFRBRow FWD', warmup=25, rep=REP)
            print('-' * 40)
        return MFRBRow.forward_(ctx, *args, **kwargs)
    

    @staticmethod
    def backward(ctx, do):
        REP = get_rep()
        if REP > 0:
            print('-' * 40)
            print(f"{__class__.__name__} BWD Profiling")
            profile(lambda: MFRBRow.backward_(ctx, do), 'Triton MFRBRow BWD', warmup=25, rep=REP)
            print('-' * 40)
        return MFRBRow.backward_(ctx, do)
    

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
            query, key, value, o, softmax_lse
        )
        
        ctx.block_size_M = block_size_M
        ctx.block_size_N = block_size_N
        ctx.causal = causal
        return o

    @staticmethod
    def backward_(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
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

        _bwd_preprocess[(grid[0] * grid[1], )](
            o, do,
            delta,
            BLOCK_M=block_size_M, 
            D_HEAD=q.shape[-1],
        )

        _bwd_row_kernel[(grid[1],)](
            q, k, v,
            sm_scale,
            context_size,
            o, do,
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], 0,
            grid[0], triton.cdiv(k.shape[2], block_size_N),
            BLOCK_M=block_size_M, 
            BLOCK_N=block_size_N,
            BLOCK_DMODEL=q.shape[-1], 
            CAUSAL=causal,
            num_warps=8,
            num_stages=1,
        )

        return dq[..., :context_size, :head_dim], dk[..., :context_size, :head_dim], dv[..., :context_size, :head_dim], None, None, None, None, None, None


@triton.jit
def _bwd_row_re_kernel(
    Q, K, V, 
    sm_scale, context_size,
    Out, DO,
    DQ, DK, DV,
    L,
    D, # [BATCH, N_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk, 
    Z, H, N_CTX, P_SEQ,
    num_block_q, num_block_kv,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
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

    # num_block_kv_curr = (start_m * BLOCK_M + BLOCK_N - 1) // BLOCK_N
    # for start_n in range(0, num_block_kv_curr):
    for start_n in range(0, num_block_kv):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # k and v stay in SRAM throughout
        k_ptrs  = K  + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dv_ptrs = DV + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :])
            qk = tl.where(causal_mask & m_mask, float(0.), float("-inf"))
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        tl.atomic_add(dv_ptrs, tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do))

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - D_i[:, None]
        dp += tl.dot(do, tl.trans(v))
        
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        tl.atomic_add(dk_ptrs, tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q))

        # compute dq
        dq += tl.dot(ds.to(Q.dtype.element_ty), k)

    dq_ptrs = DQ + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    tl.store(dq_ptrs, dq)

class MFRBRowRe(torch.autograd.Function):
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
            profile(lambda: MFRBRowRe.forward_(ctx, *args, **kwargs), 'Triton MFRBRow FWD', warmup=25, rep=REP)
            print('-' * 40)
        return MFRBRowRe.forward_(ctx, *args, **kwargs)
    

    @staticmethod
    def backward(ctx, do):
        REP = get_rep()
        if REP > 0: 
            print(f"{__class__.__name__} BWD Profiling")
            profile(lambda: MFRBRowRe.backward_(ctx, do), 'Triton MFRBRow BWD', warmup=25, rep=REP)
            print('-' * 40)
        return MFRBRowRe.backward_(ctx, do)


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
            query, key, value, o, softmax_lse
        )
        
        ctx.block_size_M = block_size_M
        ctx.block_size_N = block_size_N
        ctx.causal = causal
        return o

    @staticmethod
    def backward_(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
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

        _bwd_preprocess[(grid[0] * grid[1], )](
            o, do,
            delta,
            BLOCK_M=block_size_M, 
            D_HEAD=q.shape[-1],
        )

        _bwd_row_re_kernel[(grid[0], grid[1],)](
            q, k, v,
            sm_scale,
            context_size,
            o, do,
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], 0,
            grid[0], triton.cdiv(k.shape[2], block_size_N),
            BLOCK_M=block_size_M, 
            BLOCK_N=block_size_N,
            BLOCK_DMODEL=q.shape[-1], 
            CAUSAL=causal,
            num_warps=8,
            num_stages=1,
        )

        return dq[..., :context_size, :head_dim], dk[..., :context_size, :head_dim], dv[..., :context_size, :head_dim], None, None, None, None, None, None

