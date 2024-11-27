import torch
import triton
import triton.language as tl

from fwd_utils import _triton_mixed_sparse_attention, pad_tensor

class MFCB(torch.autograd.Function):
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

        query, key, value = (
            pad_tensor(query, context_size, head_dim, block_size_M), 
            pad_tensor(key, context_size, head_dim, block_size_M), 
            pad_tensor(value, context_size, head_dim, block_size_M)
        )

        # the input version of Q, K and V are padded to be divisible by BLOCK_SIZE_M
        # the output and softmax_lse are also padded
        sm_scale = head_dim ** -0.5

        o, softmax_lse = _triton_mixed_sparse_attention(
            query, key, value, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, context_size, head_dim,
            block_size_M, block_size_N,
        )

        ctx.save_for_backward(
            query, key, value, o, softmax_lse,
        )
        
        ctx.block_size_M = block_size_M
        ctx.block_size_N = block_size_N
        return o
    
    @staticmethod
    def backward(ctx, grad_o):
        # softmax_lse: [batch_size, num_heads, context_size]
        q, k, v, o, softmax_lse = ctx.saved_tensors

        context_size, head_dim = ctx.context_size, ctx.head_dim
        sm_scale = 1.0 / (head_dim ** 0.5)

        s = q @ k.transpose(-1, -2) * sm_scale
        causal_mask = torch.tril(
            torch.ones((q.shape[-2], q.shape[-2]), dtype=torch.bool, device='cuda'), 
            diagonal=0)
        s = torch.where(causal_mask, s, torch.full_like(s, float('-inf')))
        p = torch.nn.functional.softmax(s, dim=-1)
        
        dv = p.transpose(-1, -2) @ grad_o
        dp = grad_o @ v.transpose(-1, -2)

        # ds = p * (dp - torch.sum(dp * p, dim=-1, keepdim=True))
        ds = p * (dp - torch.sum(grad_o * o, dim=-1, keepdim=True))

        dq = ds @ k * sm_scale
        dk = ds.transpose(-1, -2) @ q * sm_scale

        return dq[:, :, :context_size, :head_dim], dk[:, :, :context_size, :head_dim], dv[:, :, :context_size, :head_dim], None, None, None, None, None, None


class MFCBMask(torch.autograd.Function):
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

        query, key, value = (
            pad_tensor(query, context_size, head_dim, block_size_M), 
            pad_tensor(key, context_size, head_dim, block_size_M), 
            pad_tensor(value, context_size, head_dim, block_size_M)
        )

        # the input version of Q, K and V are padded to be divisible by BLOCK_SIZE_M
        # the output and softmax_lse are also padded
        sm_scale = head_dim ** -0.5

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
        return o
    
    @staticmethod
    def backward(ctx, grad_o):
        # softmax_lse: [batch_size, num_heads, context_size]
        (
            q, k, v, o, softmax_lse, 
            block_count, block_offset, column_count, column_index,
        ) = ctx.saved_tensors
        context_size, head_dim = ctx.context_size, ctx.head_dim
        block_size_M, block_size_N = ctx.block_size_M, ctx.block_size_N
        num_blocks_q = context_size // block_size_M
        num_blocks_k = context_size // block_size_N

        # Convert the indices to be mask
        sparse_mask = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[-2], q.shape[-2]), 
            dtype=torch.bool, device='cuda'
        )
        print(f"sparse_mask before update\n: {sparse_mask}")

        print(f"block_count\n: {block_count}")
        print(f"block_offset\n: {block_offset}")
        print(f"column_count\n: {column_count}")
        print(f"column_index\n: {column_index}")
        for batch_idx in range(q.shape[0]):
            for head_idx in range(q.shape[1]):
                for block_idx in range(num_blocks_q):
                    rows = range(block_idx * block_size_M, min((block_idx+1) * block_size_M, context_size))
                    block_cnt = block_count[batch_idx, head_idx, block_idx]
                    block_off = block_offset[batch_idx, head_idx, block_idx, :]
                    for block_idx in range(block_cnt):
                        start_n = block_off[block_idx]
                        for row in rows:
                            for col_idx in range(start_n, min(start_n+block_size_N, context_size)):
                                sparse_mask[batch_idx, head_idx, row, col_idx] = True
                    
                    column_cnt = column_count[batch_idx, head_idx, block_idx]
                    column_idx = column_index[batch_idx, head_idx, block_idx, :]
                    columns = column_idx[:column_cnt].cpu().numpy()

                    print(f"batch_idx: {batch_idx}, head_idx: {head_idx}, rows: {rows}, columns: {columns}")
                    for row in rows:
                        for col in columns:
                            sparse_mask[batch_idx, head_idx, row, col] = True
        print(f"sparse_mask\n: {sparse_mask}")

        context_size, head_dim = ctx.context_size, ctx.head_dim
        sm_scale = 1.0 / (head_dim ** 0.5)

        s = torch.zeros((q.shape[0], q.shape[1], q.shape[-2], q.shape[-2]), device='cuda', dtype=torch.float32)
        causal_mask = torch.tril(
            torch.ones(
                (q.shape[0], q.shape[1], q.shape[-2], q.shape[-2]), dtype=torch.bool, device='cuda'
            ), diagonal=0
        )
        s = torch.where(causal_mask & sparse_mask, s, torch.full_like(s, float('-inf')))

        s += q @ k.transpose(-1, -2) 
        s *= sm_scale
        p = torch.nn.functional.softmax(s, dim=-1)

        dv = torch.zeros(v.shape, device='cuda', dtype=torch.float32)
        dv += p.transpose(-1, -2).to(q.dtype) @ grad_o

        dp = grad_o @ v.transpose(-1, -2)

        # ds = p * (dp - torch.sum(dp * p, dim=-1, keepdim=True))
        ds = p * (dp - torch.sum(grad_o * o, dim=-1, keepdim=True))

        dq = torch.zeros(q.shape, device='cuda', dtype=torch.float32)
        dq = ds.to(q.dtype) @ k * sm_scale

        dk = torch.zeros(k.shape, device='cuda', dtype=torch.float32)
        dk += ds.transpose(-1, -2).to(q.dtype) @ q * sm_scale

        return dq[:, :, :context_size, :head_dim], dk[:, :, :context_size, :head_dim], dv[:, :, :context_size, :head_dim], None, None, None, None, None, None
