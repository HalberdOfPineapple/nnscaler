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
from mfmb import MFMB, MFMBTorch
from fa_ref import attention as RefAttention
from fa_ref_2 import attention as RefAttention2
from sparse_attn_ref import sparse_multi_head_attention_backward_reference

from fa_attn_test import gen_block_indices, attn_by_mode

# ATOL, RTOL = 1e-2, 1e-2
# ATOL, RTOL = 5e-2, 5e-2
ATOL, RTOL = 1e-1, 1e-1
MODE: str
CONFIGS: Dict

SAVE_PATH = os.path.join(os.path.dirname(__file__), "perf")
os.makedirs(SAVE_PATH, exist_ok=True)


BATCH, N_HEADS, HEAD_DIM = 1, 1, 96
VERTICAL_SIZE, SLASH_SIZE = 100, -1
DISABLE_BWD = False

def init_configs():
    global MODE, BATCH, N_HEADS, HEAD_DIM, VERTICAL_SIZE, SLASH_SIZE, DISBALE_BWD

    configs = []
    for causal in [True, False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],

                # line arg represents the argument differentiating lines in the plot, 
                # here we choose "mode" for different plots for different ATTN implementations
                line_arg="mode", 
                line_vals=['mfmb', 'FA', 'ref2'],
                line_names=['MFMB', "FA-Library", "Triton"],

                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"{MODE}-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-causal={causal}-BWD={not DISABLE_BWD}",
                args={
                    "BATCH": BATCH,
                    "H": N_HEADS,
                    "vertical_size": VERTICAL_SIZE,
                    "slash_size": SLASH_SIZE,
                    "HEAD_DIM": HEAD_DIM,
                    "causal": causal,
                    "disable_bwd": DISBALE_BWD,
                },
            )
        )
    
    global CONFIGS
    CONFIGS = configs

def bench_flash_attention():
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

        ms = triton.testing.do_bench(fn)
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5

        return total_flops * 1e-12 / (ms * 1e-3)

    return bench_flash_attention_


def main(args):
    torch.manual_seed(0)
    global MODE
    MODE = args.mode
    causal: bool = not args.disable_causal

    global N_HEADS, HEAD_DIM
    N_HEADS = args.num_heads
    HEAD_DIM = 96

    global VERTICAL_SIZE, SLASH_SIZE
    VERTICAL_SIZE = 100
    if args.slash_size > 0:
        SLASH_SIZE = args.slash_size

    global DISBALE_BWD
    DISBALE_BWD = args.disable_bwd

    print('-' * 50)
    print(f"Mode: {MODE}")
    print(f"Disable Backward: {DISBALE_BWD}")
    print(f"Num Heads: {N_HEADS}")
    print(f"Head Dim: {HEAD_DIM}")
    print(f"Causal: {causal}")
    print('-' * 20)
    print(f"Vertical Size: {VERTICAL_SIZE}")
    print(f"Slash Size: {SLASH_SIZE}")
    print('-' * 20)
    print(f"SAVE_PATH = {SAVE_PATH}")
    print(f"ATOL = {ATOL}, RTOL = {RTOL}")
    

    init_configs()
    bench_func = bench_flash_attention()
    bench_func.run(save_path=SAVE_PATH, print_data=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', "--disable_bwd", action="store_true")
    parser.add_argument('-m', '--mode', type=str, default='naive')
    parser.add_argument('-nh', '--num_heads', type=int, default=1)
    parser.add_argument('-n', '--context_size', type=int, default=256)
    parser.add_argument('-c', '--disable_causal', action='store_true')
    parser.add_argument('-t', '--disable_val_equal', action='store_true')
    parser.add_argument('-r', '--rep', type=int, default=0)
    parser.add_argument('-s', '--slash_size', type=int, default=0)
    args = parser.parse_args()
    main(args)