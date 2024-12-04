import os
import sys
import math
import torch
import argparse
import functools
from typing import Callable, Dict
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import _flash_attn_forward

import triton
import triton.language as tl

from fwd_utils import pad_tensor, set_rep

from naive import NaiveAttn
from mfcb import MFCB, MFCBMask
from mfrb import MFRB, MFRBRow, MFRBRowRe
from mfmb import MFMB, MFMBTorch, MFMBDT
from fa_ref import attention as RefAttention
from fa_ref_2 import attention as RefAttention2
from sparse_attn_ref import sparse_multi_head_attention_backward_reference

from fa_attn_test import gen_block_indices, attn_by_mode

# ATOL, RTOL = 1e-2, 1e-2
# ATOL, RTOL = 5e-2, 5e-2
ATOL, RTOL = 1e-1, 1e-1
REP=100
MODE: str
CONFIGS: Dict

SAVE_PATH = os.path.join(os.path.dirname(__file__), "perf")
os.makedirs(SAVE_PATH, exist_ok=True)


BATCH, N_HEADS, HEAD_DIM = 1, 1, 96
VERTICAL_SIZE, SLASH_SIZE = 100, -1
DISABLE_BWD = False
BWD_ONLY = False
RETURN_TIME = False
ONLY_CAUSAL = True



def init_configs():
    global MODE, BATCH, N_HEADS, HEAD_DIM, VERTICAL_SIZE, SLASH_SIZE, DISABLE_BWD, RETURN_TIME


    print(f"DISABLE_BWD = {DISABLE_BWD}, BWD_ONLY = {BWD_ONLY}", end=', ')
    if not DISABLE_BWD and not BWD_ONLY:
        run_mode = "FULL"
    elif BWD_ONLY:
        run_mode = "BWD"
    else:
        run_mode = "FWD"
    print(f"run_mode = {run_mode}")

    causal_choices = [True] if ONLY_CAUSAL else [True, False]

    configs = []
    for causal in causal_choices:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2 ** i for i in range(10, 20 + 1)],

                # line arg represents the argument differentiating lines in the plot, 
                # here we choose "mode" for different plots for different ATTN implementations
                line_arg="mode", 
                line_vals=[MODE.lower(), 'FA', 'ref2'],
                line_names=[MODE, "FA-Library", "Triton"],

                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="MS" if RETURN_TIME else "TFLOPS",
                plot_name=
                    f"{MODE}-{'ms' if RETURN_TIME else 'tflops'}-" \
                    f"batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-v{VERTICAL_SIZE}" \
                    f"-s{SLASH_SIZE}-causal={causal}-run_mode={run_mode}",
                args={
                    "BATCH": BATCH,
                    "H": N_HEADS,
                    "vertical_size": VERTICAL_SIZE,
                    "slash_size": SLASH_SIZE,
                    "HEAD_DIM": HEAD_DIM,
                    "causal": causal,
                    "disable_bwd": DISABLE_BWD,
                },
            )
        )
    
    global CONFIGS
    CONFIGS = configs

def bench_flash_attention(bwd_only: bool=False):
    global CONFIGS, MODE

    @triton.testing.perf_report(CONFIGS)
    def bench_flash_attention_(
        BATCH, H, N_CTX, HEAD_DIM, 
        vertical_size, slash_size,
        causal, 
        mode,
        disable_bwd,
        device="cuda"
    ):
        dtype = torch.float16
        slash_size = N_CTX if slash_size <= 0 else slash_size

        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

        print(f"vertical_size = {vertical_size}, slash_size = {slash_size}")
        block_count, block_offset, column_count, column_index, seqlens = gen_block_indices(
            q, k, v, N_CTX, vertical_size, slash_size, HEAD_DIM
        )

        def fn():
            o = attn_by_mode(
                mode, q, k, v, N_CTX, HEAD_DIM,
                block_count, block_offset, column_count, column_index, 
                seqlens, causal
            )
            if not disable_bwd:
                torch.cuda.synchronize()
                loss = torch.square(o).sum(dtype=torch.float64)
                loss.backward()
            return o

        ms = triton.testing.do_bench(fn, rep=REP)
        if RETURN_TIME:
            return ms
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        
        if not disable_bwd:
            total_flops *= 3.5

        return total_flops * 1e-12 / (ms * 1e-3)

    
    @triton.testing.perf_report(CONFIGS)
    def bench_flash_attention_bwd(
        BATCH, H, N_CTX, HEAD_DIM, 
        vertical_size, slash_size,
        causal, 
        mode,
        disable_bwd,
        device="cuda"
    ):
        dtype = torch.float16
        slash_size = N_CTX if slash_size <= 0 else slash_size

        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

        print(f"vertical_size = {vertical_size}, slash_size = {slash_size}")
        block_count, block_offset, column_count, column_index, seqlens = gen_block_indices(
            q, k, v, N_CTX, vertical_size, slash_size, HEAD_DIM
        )

        o = attn_by_mode(
                mode, q, k, v, N_CTX, HEAD_DIM,
                block_count, block_offset, column_count, column_index, 
                seqlens, causal
            )
        def fn():
            torch.cuda.synchronize()
            loss = torch.square(o).sum(dtype=torch.float64)
            loss.backward(retain_graph=True)
            return o

        ms = triton.testing.do_bench(fn, rep=REP)
        if RETURN_TIME: 
            return ms

        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 5 * flops_per_matmul
        if causal:
            total_flops *= 0.5

        return total_flops * 1e-12 / (ms * 1e-3)

    if not bwd_only:    
        return bench_flash_attention_
    else:
        return bench_flash_attention_bwd


def main(args):
    torch.manual_seed(0)
    global MODE
    MODE = args.mode

    global N_HEADS, HEAD_DIM
    N_HEADS = args.num_heads
    HEAD_DIM = 96

    global VERTICAL_SIZE, SLASH_SIZE
    VERTICAL_SIZE = args.vertical_size
    if args.slash_size > 0:
        SLASH_SIZE = args.slash_size

    if args.fwd_only and args.bwd_only:
        raise ValueError("Cannot enable both fwd_only and bwd_only")
    
    global DISABLE_BWD
    DISABLE_BWD = args.fwd_only

    global BWD_ONLY
    BWD_ONLY = args.bwd_only

    global RETURN_TIME
    RETURN_TIME = args.use_ms

    global ONLY_CAUSAL
    ONLY_CAUSAL = not args.enable_noncausal

    print('-' * 50)
    print(f"Mode: {MODE}")
    print(f"FWD Only: {DISABLE_BWD}")
    print(f"BWD Only: {BWD_ONLY}")
    print(f"Num Heads: {N_HEADS}")
    print(f"Head Dim: {HEAD_DIM}")
    print(f"Only Causal: {ONLY_CAUSAL}")
    print(f"Metric: {'MS' if RETURN_TIME else 'TFLOPS'}")
    print('-' * 20)
    print(f"Vertical Size: {VERTICAL_SIZE}")
    print(f"Slash Size: {SLASH_SIZE}")
    print('-' * 20)
    print(f"SAVE_PATH = {SAVE_PATH}")
    print(f"ATOL = {ATOL}, RTOL = {RTOL}")
    

    init_configs()
    bench_func = bench_flash_attention(bwd_only=args.bwd_only)
    bench_func.run(save_path=SAVE_PATH, print_data=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='naive')
    parser.add_argument('-nh', '--num_heads', type=int, default=1)
    parser.add_argument('-n', '--context_size', type=int, default=256)
    
    parser.add_argument('-s', '--slash_size', type=int, default=0)
    parser.add_argument('-v', '--vertical_size', type=int, default=100)

    parser.add_argument('-f', '--fwd_only', action='store_true')
    parser.add_argument('-b', '--bwd_only', action='store_true')

    parser.add_argument('--use_ms', action='store_true')
    parser.add_argument('-enc', '--enable_noncausal', action='store_true')
    parser.add_argument('-r', '--rep', type=int, default=0)

    args = parser.parse_args()

    main(args)