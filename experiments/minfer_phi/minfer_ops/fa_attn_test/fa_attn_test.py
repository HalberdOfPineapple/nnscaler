import sys
import math
import torch
import argparse
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import _flash_attn_forward

import triton
import triton.language as tl

from fwd_utils import pad_tensor, set_rep

from naive import NaiveAttn
from mfcb import MFCB, MFCBMask
from mfrb import MFRB, MFRBRow, MFRBRowRe
from mfmb import MFMB, MFMBTorch
from fa_ref import attention as RefAttention
from fa_ref_2 import attention as RefAttention2
from sparse_attn_ref import sparse_multi_head_attention_backward_reference


# ATOL, RTOL = 1e-2, 1e-2
# ATOL, RTOL = 5e-2, 5e-2
ATOL, RTOL = 1e-1, 1e-1

def gen_block_indices(
    q: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    q_len: int, vertical_size: int, slash_size: int, head_dim: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    from minference.cuda import convert_vertical_slash_indexes
    from minference.modules.minference_forward import (
        LAST_Q_MASK, sum_all_diagonal_matrix
    )
    batch_size, num_heads, context_size, head_dim = q.shape

    vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
    last_q = min(64, q_len)

    with torch.no_grad():
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
    
    return block_count, block_offset, column_count, column_index, seqlens

def attn_by_mode(
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context_size: int,
        head_dim: int,
        block_count: torch.Tensor,
        block_offset: torch.Tensor,
        column_count: torch.Tensor,
        column_index: torch.Tensor,
        seqlens: torch.Tensor,
        causal: bool = True,
):
    if mode == "naive":
        o = NaiveAttn.apply(q, k, v, context_size, head_dim)
    elif mode == "mfcb":
        o = MFCB.apply(
            q, k, v,
            block_count, block_offset, column_count, column_index,
            seqlens
        )
        o = o[..., :, :context_size, :head_dim]
    elif mode == "mfcb_mask":
        o = MFCBMask.apply(
            q, k, v,
            block_count, block_offset, column_count, column_index,
            seqlens
        )
        o = o[..., :, :context_size, :head_dim]
    elif mode == "FA":
        o = flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
        )
        o = o.transpose(1, 2)
    elif mode == "mfrb":
        o = MFRB.apply(
            q, k, v,
            block_count, block_offset, column_count, column_index,
            seqlens
        )
        o = o[..., :, :context_size, :head_dim]
        # o = mfrb(q, k, v, seqlens, context_size, head_dim)
    elif mode == "mfrb_row":
        o = MFRBRow.apply(
            q, k, v,
            block_count, block_offset, column_count, column_index,
            seqlens,
            causal,
        )
        o = o[..., :, :context_size, :head_dim]
    elif mode == "mfrb_row_re":
        o = MFRBRowRe.apply(
            q, k, v,
            block_count, block_offset, column_count, column_index,
            seqlens,
            causal,
        )
        o = o[..., :, :context_size, :head_dim]
    elif mode == "mfmb":
        o = MFMB.apply(
            q, k, v,
            block_count, block_offset, column_count, column_index,
            seqlens,
            causal,
        )
        o = o[..., :, :context_size, :head_dim]
    elif mode == "mfmb_torch":
        o = MFMBTorch.apply(
            q, k, v,
            block_count, block_offset, column_count, column_index,
            seqlens,
            causal,
        )
        o = o[..., :, :context_size, :head_dim]
    elif mode == "ref":
        q_pad = pad_tensor(q, context_size, head_dim, 64)
        k_pad = pad_tensor(k, context_size, head_dim, 64)
        v_pad = pad_tensor(v, context_size, head_dim, 64)
        o = RefAttention(
            q_pad, k_pad, v_pad,
            True,
            1.0 / (head_dim ** 0.5),
        )
        o = o[..., :, :context_size, :head_dim]
    elif mode == "ref2":
        q_pad = pad_tensor(q, context_size, head_dim, 64)
        k_pad = pad_tensor(k, context_size, head_dim, 64)
        v_pad = pad_tensor(v, context_size, head_dim, 64)
        o = RefAttention2(
            q_pad, k_pad, v_pad,
            True,
            1.0 / (head_dim ** 0.5),
        )
        o = o[..., :, :context_size, :head_dim]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return o

def main(args):
    torch.manual_seed(0)
    mode: str = args.mode
    ref_mode: str = args.ref_mode
    causal: bool = not args.disable_causal

    set_rep(args.rep)

    context_size = args.context_size
    num_heads = args.num_heads
    head_dim = 96

    vertical_size = 100
    slash_size = context_size if args.slash_size <= 0 else args.slash_size

    print('-' * 50)
    print(f"Mode: {mode}")
    print(f"Context Size: {context_size}")
    print(f"Num Heads: {num_heads}")
    print(f"Head Dim: {head_dim}")
    print(f"Causal: {causal}")
    print('-' * 20)
    print(f"Vertical Size: {vertical_size}")
    print(f"Slash Size: {slash_size}")
    print('-' * 20)
    print(f"ATOL = {ATOL}, RTOL = {RTOL}")
    # print('-' * 50)

    # q = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    # k = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    # v = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)

    factor = 1.
    q = torch.ones((1, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    k = torch.ones((1, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    v = torch.ones((1, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)

    # --------------------------------------------------------------------------------
    # Generate block indices
    block_count, block_offset, column_count, column_index, seqlens = gen_block_indices(
        q, k, v, context_size, vertical_size, slash_size, head_dim
    )
    # print('-' * 50)
    # print(f"Block Count:\n{block_count}")
    # print(f"Block Offset:\n{block_offset}")
    # print(f"Column Count:\n{column_count}")
    # print(f"Column Index:\n{column_index}")

    # --------------------------------------------------------------------------------
    # Standard FA Library for reference
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o_ref = attn_by_mode(
        ref_mode, q, k, v, context_size, head_dim,
        block_count, block_offset, column_count, column_index, seqlens,
        causal=causal,
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
    o_ref.grad.zero_()
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # --------------------------------------------------------------------------------
    # Custom Attention
    o = attn_by_mode(
        mode,
        q, k, v, context_size, head_dim,
        block_count, block_offset, column_count, column_index, seqlens,
        causal=causal,
    )
                     
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
    print(f"q_grad shape: {q_grad.shape}")


    # --------------------------------------------------------------------------------
    print('-' * 80)
    print(f"Comparing {ref_mode} (ref) and {mode} attention")
    for head_idx in range(num_heads):
        print('-' * 40)
        output_close = torch.allclose(o[0, head_idx, :, :], o_ref[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        output_grad_close = torch.allclose(o_grad[0, head_idx, :, :], o_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        q_grad_close = torch.allclose(q_grad[0, head_idx, :, :], q_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        k_grad_close = torch.allclose(k_grad[0, head_idx, :, :], k_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        v_grad_close = torch.allclose(v_grad[0, head_idx, :, :], v_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)

        if not output_close or args.disable_val_equal: 
            print('-' * 20)
            if not output_close: print(f"Head {head_idx} output is not close")
            print(f"Output:\n{o[0, head_idx, :, :]}")
            print(f"Output Ref:\n{o_ref[0, head_idx, :, :]}")

        if not output_grad_close or args.disable_val_equal: 
            print('-' * 20)
            if not output_grad_close: print(f"Head {head_idx} output grad is not close")
            print(f"Output:\n{o_grad[0, head_idx, :, :]}")
            print(f"Output Ref:\n{o_ref_grad[0, head_idx, :, :]}")

        if not q_grad_close or args.disable_val_equal: 
            print('-' * 20)
            if not q_grad_close: print(f"Head {head_idx} q grad is not close")
            print(f"Q Grad:\n{q_grad[0, head_idx, :, :]}")
            print(f"Q Grad Ref:\n{q_ref_grad[0, head_idx, :, :]}")
        
        if not k_grad_close or args.disable_val_equal: 
            print('-' * 20)
            if not k_grad_close: print(f"Head {head_idx} k grad is not close")
            print(f"K Grad:\n{k_grad[0, head_idx, :, :]}")
            print(f"K Grad Ref:\n{k_ref_grad[0, head_idx, :, :]}")

        if not v_grad_close or args.disable_val_equal: 
            print('-' * 20)
            if not v_grad_close: print(f"Head {head_idx} v grad is not close")
            print(f"V Grad:\n{v_grad[0, head_idx, :, :]}")
            print(f"V Grad Ref:\n{v_ref_grad[0, head_idx, :, :]}")
    
    # --------------------------------------------------------------------------------
    # print('-' * 80)
    # print(f"Comparing sparse reference and {mode} attention")
    # recovered_mask = torch.ones((context_size, context_size), dtype=torch.float16, device='cuda')
    # q_grad_sr, k_grad_sr, v_grad_sr = sparse_multi_head_attention_backward_reference(
    #     o_grad, o, q, k, v, 
    #     recovered_mask,
    #     transposed=True,
    # )
    
    # for head_idx in range(num_heads):
    #     print('-' * 40)
    #     q_grad_close = torch.allclose(q_grad[0, head_idx, :, :], q_grad_sr[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
    #     k_grad_close = torch.allclose(k_grad[0, head_idx, :, :], k_grad_sr[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
    #     v_grad_close = torch.allclose(v_grad[0, head_idx, :, :], v_grad_sr[0, head_idx, :, :], atol=ATOL, rtol=RTOL)

    #     if not q_grad_close or args.disable_val_equal: 
    #         print('-' * 20)
    #         if not q_grad_close: print(f"Head {head_idx} q grad is not close")
    #         print(f"Q Grad:\n{q_grad[0, head_idx, :, :]}")
    #         print(f"Q Grad Ref:\n{q_grad_sr[0, head_idx, :, :]}")
        
    #     if not k_grad_close or args.disable_val_equal: 
    #         print('-' * 20)
    #         if not k_grad_close: print(f"Head {head_idx} k grad is not close")
    #         print(f"K Grad:\n{k_grad[0, head_idx, :, :]}")
    #         print(f"K Grad Ref:\n{k_grad_sr[0, head_idx, :, :]}")

    #     if not v_grad_close or args.disable_val_equal: 
    #         print('-' * 20)
    #         if not v_grad_close: print(f"Head {head_idx} v grad is not close")
    #         print(f"V Grad:\n{v_grad[0, head_idx, :, :]}")
    #         print(f"V Grad Ref:\n{v_grad_sr[0, head_idx, :, :]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='naive')
    parser.add_argument('-rm', '--ref_mode', type=str, default='FA')
    parser.add_argument('-nh', '--num_heads', type=int, default=1)
    parser.add_argument('-n', '--context_size', type=int, default=256)
    parser.add_argument('-c', '--disable_causal', action='store_true')
    parser.add_argument('-t', '--disable_val_equal', action='store_true')
    parser.add_argument('-r', '--rep', type=int, default=0)
    parser.add_argument('-s', '--slash_size', type=int, default=0)
    args = parser.parse_args()
    main(args)