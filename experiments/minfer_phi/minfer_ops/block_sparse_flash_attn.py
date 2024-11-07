import torch
from nnscaler.graph.parser.register import register_op
from minference.ops.block_sparse_flash_attention import block_sparse_attention


class CustomBSAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk):
        ctx.save_for_backward(q, k, v, topk)
        return block_sparse_attention(q, k, v, topk)
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement backward pass
        return grad_output.clone(), grad_output.clone(), grad_output.clone(), None


def bs_attn_forward(q, k, v, topk):
    return CustomBSAttention.apply(q, k, v, topk)


register_op('b num_head^ l^ hd^, b num_head^ l^ hd^, b num_head^ l^ hd^ -> b num_head^ l^ hd^')(bs_attn_forward)