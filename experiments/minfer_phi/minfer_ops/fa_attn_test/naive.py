import torch 

class NaiveAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, context_size, head_dim):
        sm_scale = 1.0 / (head_dim ** 0.5)
        s = q @ k.transpose(-1, -2) * sm_scale

        causal_mask = torch.tril(torch.ones((context_size, context_size), dtype=torch.bool, device='cuda'), diagonal=0)
        s = torch.where(causal_mask, s, torch.full_like(s, float('-inf')))

        p = torch.nn.functional.softmax(s, dim=-1)
        o = p @ v 

        ctx.save_for_backward(o, q, k, v)
        ctx.sm_scale = sm_scale
        return o
    
    @staticmethod
    def backward(ctx, grad_o):
        o, q, k, v = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        s = q @ k.transpose(-1, -2) * sm_scale
        causal_mask = torch.tril(torch.ones((q.shape[-2], q.shape[-2]), dtype=torch.bool, device='cuda'), diagonal=0)
        s = torch.where(causal_mask, s, torch.full_like(s, float('-inf')))
        p = torch.nn.functional.softmax(s, dim=-1)

        dv = p.transpose(-1, -2) @ grad_o
        dp = grad_o @ v.transpose(-1, -2)

        # ds = p * (dp - torch.sum(dp * p, dim=-1, keepdim=True))
        ds = p * (dp - torch.sum(grad_o * o, dim=-1, keepdim=True))

        dq = ds @ k * sm_scale
        dk = ds.transpose(-1, -2) @ q * sm_scale

        return dq, dk, dv, None, None