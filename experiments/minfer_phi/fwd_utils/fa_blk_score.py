import math
import torch

import triton
import triton.language as tl

from flash_attn import flash_attn_func

def pad_tensor(x):
    head_dim = x.shape[-1]
    dim_pad = 2 ** math.ceil(math.log2(head_dim)) - head_dim
    x_pad = torch.nn.functional.pad(x, [0, dim_pad, 0, 0, 0, 0, 0, 0])
    return x_pad


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
#    key=[],
# )
@triton.jit
def triton_attn_score_fwd_kernel(
    Q, K, V, sm_scale,
    P, M, L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_pz, stride_ph, stride_pm, stride_pn,
    stride_lz, stride_lh, stride_lm,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to p, m and l
    p_ptrs = P + (off_hz // H) * stride_pz + (off_hz % H) * stride_ph + offs_m * stride_pm
    m_ptrs = M + (off_hz // H) * stride_pz + (off_hz % H) * stride_ph + offs_m * stride_pm
    l_ptrs = L + (off_hz // H) * stride_lz + (off_hz % H) * stride_lh + offs_m * stride_lm

    # initialize m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-6
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option='zero')
    q = (q * qk_scale).to(dtype)
    # loop over k, v and update accumulator

    m_mask = offs_m < N_CTX # [BLOCK_M]
    for start_n in range(0, N_CTX, BLOCK_N):
        # n_mask = start_n + offs_n[None, :] < N_CTX
        cols = start_n + offs_n
        n_mask = cols[None, :] < N_CTX

        # -- load k, v --
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None] # [BLOCK_M, BLOCK_N]
        qk = tl.where(m_mask[:, None] & causal_mask, qk, float("-inf"))
        # qk = tl.where(n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)

        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- write back averaged p and m_i_new --
        sum_p = tl.sum(p, 1)
        tl.store(p_ptrs, sum_p, mask=m_mask) # p stores the sum of current QK following column dimension divided by the current max (m_i_new)
        tl.store(m_ptrs, m_i_new, mask=m_mask) # m stores the current max (m_i_new)

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + sum_p
        m_i = m_i_new

        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        p_ptrs += stride_pn
        m_ptrs += stride_pn

    # write back O
    acc /= l_i[:, None]
    O_block_ptr = tl.make_block_ptr(
        base=Out + qo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(dtype), boundary_check=(1, 0))

    # write back L
    m_i += tl.math.log2(l_i)
    tl.store(l_ptrs, m_i)


def triton_attn_score_forward(
    q: torch.Tensor,            # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,            # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,            # [BATCH, N_HEADS, N_CTX, D_HEAD]
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    # shape constraints
    B, Nq, H, Lq = q.shape
    _, Nk, _, Lk = k.shape
    Lv = v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    Tq = triton.cdiv(Nq, block_size_M)
    Tk = triton.cdiv(Nk, block_size_N)
    grid = (Tq, B * H, 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    num_stages = 2

    # allocate memory for outputs
    o = torch.zeros_like(q)
    p = torch.zeros(size=(B, H, Tq * block_size_M, Tk), dtype=torch.float32, device=q.device)
    m = torch.zeros_like(p)
    l = torch.zeros(size=(B, H, Tq * block_size_M), dtype=torch.float32, device=q.device)

    with torch.cuda.device(q.device):
        # launch the kernel
        triton_attn_score_fwd_kernel[grid](
            q, k, v, sm_scale,
            p, m, l,
            o,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),
            p.stride(0), p.stride(1), p.stride(2), p.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            B, H, Nq,
            BLOCK_M=block_size_M, BLOCK_N=block_size_N,
            BLOCK_DMODEL=Lk,
            dtype=dtype,
            num_warps=4, 
            num_stages=num_stages,
        )

    # calculate attention score
    p *= torch.exp2(m - l[:, :, :, None]) # the block-wise sum multiplies the correpsonding maximum to recover the original column-wise sum of block QK and divide the complete row sum
    p = p.reshape(B, H, Tq, block_size_M, Tk).mean(dim=-2) # average the block-wise sum along the row dimension to get the average pool of QK within each block
    return o, p


def flash_attn_forward(q, k, v, sm_scale) -> torch.Tensor:
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=False,
    )


def torch_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
    mask: torch.Tensor,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.einsum(f'bmhk, bnhk -> bhmn', query, key) * sm_scale
    if mask is not None:
        p = p.where(mask, -torch.inf)
    p_max = p.max(-1, keepdim=True).values
    p_max = torch.where(p_max < 0, 0.0, p_max)
    p_exp = torch.exp(p - p_max)
    s = p_exp / (p_exp.sum(-1, keepdim=True) + 1e-6)
    out = torch.einsum(f'bhmn, bnhk -> bmhk', s, value)

    B, Nq, H, _ = query.shape
    _, Nk, _, _ = key.shape
    Pq = block_size_M - 1 - (Nq - 1) % block_size_M
    Pk = block_size_N - 1 - (Nk - 1) % block_size_N
    Tq = (Nq + Pq) // block_size_M
    Tk = (Nk + Pk) // block_size_N
    attn_score = torch.nn.functional.pad(s, (0, Pk, 0, Pq, 0, 0, 0, 0), value=0.0).to(torch.float32)
    attn_score = attn_score.reshape((B, H, Tq, block_size_M, Tk, block_size_N)).swapaxes(3, 4).sum(dim=-1).mean(dim=-1)
    return out, attn_score


def triton_flash_attn_with_block_score(
    query: torch.Tensor,        # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,        # [BATCH, N_HEADS, N_CTX, D_HEAD]
    sm_scale: float = None,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    B, Nq, H, Lq = query.shape
    _, Nk, _, _ = key.shape
    if sm_scale is None:
        sm_scale = Lq ** -0.5

    query, key, value = (
        pad_tensor(query), 
        pad_tensor(key), 
        pad_tensor(value)
    )

    with torch.no_grad():
        if (Nq > 16384 and Nk > 16384) and (B > 1 or H > 1):
            Tq = triton.cdiv(Nq, block_size_M)
            Tk = triton.cdiv(Nk, block_size_N)
            out = torch.zeros_like(query)
            score = torch.zeros(size=(B, H, Tq, Tk), dtype=torch.float32, device=query.device)
            for b in range(B):
                for h in range(H):
                    o, a = triton_attn_score_forward(
                        query[b:b+1, :, h:h+1, :],
                        key[b:b+1, :, h:h+1, :],
                        value[b:b+1, :, h:h+1, :],
                        sm_scale, block_size_M, block_size_N,
                    )
                    out[b:b+1, :, h:h+1, :] = o
                    score[b:b+1, h:h+1, :, :] = a
        else:
            out, score = triton_attn_score_forward(query, key, value, sm_scale, block_size_M, block_size_N)

    return out, score


def profile(fn, total_flops, tag, warmup=25, rep=100):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if total_flops > 0:
        gflops = total_flops / ms * 1e-9
        print(f'  {tag} : {ms:.3f} ms | {gflops:.3f} GFLOP/s')
    else:
        print(f'  {tag} : {ms:.3f} ms')


def test_flash_attention(
    dtype=torch.float16,
    device="cuda",
    batch_size=4,
    num_heads=24,
    context_size=4096,
    head_dim=64,
    block_size_M=64,
    block_size_N=64,
    torch_check=False,
):
    print('============================================================')
    print(f'SHAPE={batch_size, context_size, num_heads, head_dim}, BLOCK={block_size_M, block_size_N}')
    q = torch.randn((batch_size, context_size, num_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, context_size, num_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, context_size, num_heads, head_dim), dtype=dtype, device=device)
    sm_scale = head_dim ** -0.5
    dense_flops = 2. * batch_size * num_heads * context_size * context_size * head_dim


    triton_fn = lambda: triton_flash_attn_with_block_score(q, k, v, sm_scale, block_size_M, block_size_N)
    triton_output, attn_score = triton_fn()

    flash_fn = lambda: flash_attn_forward(q, k, v, sm_scale)
    flash_output = flash_fn()

    if torch_check:
        ref_o, ref_attn_score = torch_forward(
            q, k, v, sm_scale, mask=None, block_size_M=block_size_M, block_size_N=block_size_N
        )
        torch.testing.assert_close(triton_output, ref_o, atol=1e-3, rtol=0)
        torch.testing.assert_close(attn_score, ref_attn_score, atol=1e-3, rtol=0)
        torch.testing.assert_close(flash_output, ref_o, atol=1e-3, rtol=0)

    profile(triton_fn, dense_flops, 'triton')
    profile(flash_fn, dense_flops, ' flash')
    print('============================================================\n')


if __name__ == '__main__':
    torch.manual_seed(2024)
    test_flash_attention(batch_size=4, num_heads=24, head_dim=64, context_size=4250, block_size_M=64, block_size_N=64, torch_check=True)
    test_flash_attention(batch_size=4, num_heads=4, head_dim=64, context_size=16400, block_size_M=64, block_size_N=64, torch_check=True)
    test_flash_attention(batch_size=1, num_heads=32, head_dim=128, context_size=131072, block_size_M=64, block_size_N=64, torch_check=False)
