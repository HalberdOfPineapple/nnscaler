import os
import sys
import yaml
import json
import time
import math
import torch
import argparse
import datetime
import subprocess
import numpy as np

from datasets import load_from_disk
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union, Any, Dict
from transformers import DataCollatorForLanguageModeling
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
)

from nnscaler.cli.trainer_args import (
    DatasetConfig,
    TrainerArgs,
    ComputeConfig,
    DataloaderConfig,
    DatasetSamplerConfig,
    load_type
)
from phi3 import (
    Phi3Config as PhiConfig, Phi3Attention as PhiAttention, 
    PHI3_ATTENTION_CLASSES as PHI_ATTENTION_CLASSES, 
    apply_rotary_pos_emb, repeat_kv, _get_unpad_data
)
from modeling_modifier import NNScalerPhiFlashAttention2
from minfer_modifier_v2 import MInferAttention
from train import get_tokenizer, BaselineModel, MInferModel
from minfer_ops import gen_block_indices
from fwd_utils import triton_flash_attn_with_block_score

with open("/scratch/sync/.secrets/sas_token", "r") as f:
    SAS_TOKEN = f.read().strip()
SAS_POSTFIX = f"?{SAS_TOKEN}"

IGNORE_IDX = -100
DEVICE = 'cuda:0'

SEQ_LEN = 131072
K_BLOCK_SIZE = 4096
NUM_ATTN_BLOCKS = 10
NUM_ATTN_SAMPLES = 5

GLOBAL_BATCH_OFFSET = 111
NUM_GLOBAL_BATCHES = 5
GLOBAL_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 1

# NUM_GLOBAL_BATCHES = 2
# GLOBAL_BATCH_SIZE = 5
# MICRO_BATCH_SIZE = 1

NUM_LAYERS = 32
NUM_HEADS = 32

IGNORE_IDX = -100
STORE_DIR = '/scratch/sync/nnscaler_store'
MINFER_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
SPARSE_PATTERN_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'minfer_modules', 'configs')
SPARSE_HEAD_MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'minfer_modules', 'sparse_head_maps')

def run_cmd(cmd):
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error running command {cmd}. Exiting...")
        return 1

    return 0

def transfer_by_cp(local_path, dir='upload', source=None):
    remote_path = local_path.replace('/scratch/eval', '/scratch/sync/nnscaler_store')
    if dir == 'upload':
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        run_res = run_cmd([
            'cp', local_path, remote_path
        ])
    else:
        if source == None:
            run_res = run_cmd([
                'cp', remote_path, local_path
            ])
        else:
            run_res = run_cmd([
                'cp', source, local_path
            ])
    
    if run_res != 0:
        print(f"Error {'uploading' if dir == 'upload' else 'downloading'} {local_path}. Exiting...")
        return 1
    return 0


def merge_ckpts(args):
    log_file_path = os.path.join(
        f'/scratch/sync/nnscaler_store/{args.gpu_set}', args.expr_dir, args.expr_name,
        'checkpoints', f'{args.epoch_idx:04d}-{args.iter_idx:04d}', 'merge_ckpt.log'
    )
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)


    # --------------------------------------------------------------------------
    # Set ckpt save path and URL
    nnscaler_store_path = f"/scratch/sync/nnscaler_store/{args.gpu_set}"
    ckpt_dir = os.path.join(
        nnscaler_store_path, args.expr_dir, args.expr_name, "checkpoints", 
        f"{args.epoch_idx:04d}-{args.iter_idx:04d}",
    )
    ckpt_save_dir = os.path.join(ckpt_dir, "merged")
    ckpt_save_path = os.path.join(ckpt_save_dir, "pytorch_model.bin")
    os.makedirs(ckpt_save_dir, exist_ok=True)

    if os.path.exists(os.path.join(ckpt_dir, 'merged', 'pytorch_model.bin')) and not args.override:
        print(f"Checkpoint already exists in {ckpt_dir} and skip merging")
        return True

    # --------------------------------------------------------------------------
    print('-' * 60)
    print(f"Checkpoint merging from {ckpt_dir} to {ckpt_save_dir}...", flush=True)
    os.system(
        f"python /scratch/nnscaler/experiments/minfer_phi/ckpt_merger.py "
        f"--ckpt_dir {ckpt_dir} "
        f"--output_fname {ckpt_save_path}"
    )

    print(f"\tConverting merged checkpoint to correct format...", flush=True)
    ckpt_data = torch.load(ckpt_save_path)
    ckpt_model_data = ckpt_data.pop('model')
    ckpt_model_data = {k[6:]: v for k, v in ckpt_model_data.items()}
    torch.save(ckpt_model_data, ckpt_save_path)
    print(f"Checkpoint merged.", flush=True)

    return True

def copy_config_files(args):
    sync_dir = os.path.join(
        STORE_DIR, args.gpu_set, args.expr_dir, args.expr_name, 
        "checkpoints", f"{args.epoch_idx:04d}-{args.iter_idx:04d}",
    )
    sync_ckpt_path = os.path.join(sync_dir, 'merged', "pytorch_model.bin")
    sync_pattern_path = os.path.join(sync_dir, 'eval', "sparse_pattern.json")
    if not os.path.exists(sync_ckpt_path):
        raise ValueError(f"Merged checkpoint path {sync_ckpt_path} does not exist on Azure storage. Run merge_ckpt first.")

    if os.path.exists(sync_pattern_path) and not args.override:
        print(f"Pattern file {sync_pattern_path} already exists on Azure storage. Use -o to override.")
        return True

    local_dir = os.path.join(
        '/scratch/eval', args.gpu_set, args.expr_dir, args.expr_name, "checkpoints",
        f"{args.epoch_idx:04d}-{args.iter_idx:04d}", 
    )
    local_eval_dir = os.path.join(local_dir, "eval")
    local_merge_dir = os.path.join(local_dir, "merged")
    ckpt_output_path = os.path.join(local_merge_dir, "pytorch_model.bin")
    os.makedirs(local_merge_dir, exist_ok=True)
    os.makedirs(local_eval_dir, exist_ok=True)

    print('-' * 20)
    print(f"Copying model files to {local_merge_dir}", flush=True)
    model_name = args.model_id.split("/")[-1]
    for file in [
        "added_tokens.json", "generation_config.json", "modeling_phi3.py", 
        "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"
    ]:
        hf_file_path = os.path.join("/scratch/sync/hf_repos", model_name, file)
        local_path = os.path.join(local_merge_dir, file)
        print(f"\tCopying {hf_file_path} to {local_path}...")
        result = transfer_by_cp(
            local_path, 
            source=hf_file_path,
            dir='download',
        )
        if result != 0:
            print(f"Error copying {hf_file_path} to {local_path}. Exiting...")
            return False
    
    print('-' * 20)
    print(f"Copying config files to {local_merge_dir}", flush=True)
    expr_config_dir = os.path.join(os.getenv("NNSCALER_HOME"), "experiments", args.expr_dir, "phi3", "lc_config")
    for file in ["config.json", "configuration_phi3.py"]:
        expr_config_path = os.path.join(expr_config_dir, file)
        local_config_path = os.path.join(local_merge_dir, file)
        
        if os.path.exists(local_config_path) and not args.override:
            print(f"File {local_config_path} already exists. Use -o to override.")
            continue
        
        os.makedirs(os.path.dirname(local_config_path), exist_ok=True)
        result = run_cmd(
            [
                "cp", expr_config_path, local_config_path,
            ],
        )
        if result != 0:
            print(f"Error copying {expr_config_path} to {local_config_path}. Exiting...")
            return False

    print('-' * 20)
    print(f"Downloading checkpoint from Azure...", flush=True)
    if os.path.exists(ckpt_output_path) and not args.override:
        print(f"Checkpoint already exists in {ckpt_output_path} and skip downloading")
    else:
        result = transfer_by_cp(ckpt_output_path, dir='download')
        if result != 0: return False
        print(f"Done")
    return True


def _fix_input(input, input_dtype):
    if isinstance(input, dict):
        return {k: _fix_input(v, input_dtype) for k, v in input.items()}
    elif isinstance(input, list):
        return [_fix_input(v, input_dtype) for v in input]
    elif isinstance(input, tuple):
        return tuple(_fix_input(v, input_dtype) for v in input)
    elif isinstance(input, torch.Tensor):
        if input.is_floating_point() and input_dtype is not None:
            return input.to(input_dtype).to(DEVICE)
        else:
            return input.to(DEVICE)
    return input


def _global_batch_iterator(trainer_dataloader, update_freq, num_skip_first = 0, stage='train'):
    samples = []
    for idx, sample in enumerate(trainer_dataloader[stage]):
        if idx < num_skip_first * update_freq:
            continue
        sample = _fix_input(sample, input_dtype=torch.bfloat16)
        samples.append(sample)
        if len(samples) == update_freq:
            yield samples
            samples = []
    if samples:
        yield samples



def get_dataloader(
        model_id: str,
        scaling_factor = 1,
    ) -> DataLoader:
    trainer_dataloader: Dict[str, Optional[DataLoader]] = {'train': None, 'val': None, 'test': None}
    tokenizer = get_tokenizer(model_id)

    dataset_path = os.path.join(os.getenv("NNSCALER_LOCAL"), f"bookcorpus_phi_{SEQ_LEN}")
    dataset_disk = load_from_disk(dataset_path)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    def collate_fn(samples):
        if len(samples) == 0:
            return {}

        mini_batch = data_collator(samples)
        _mini_batch = {}

        src_tokens = mini_batch.pop('input_ids')
        seq_len = src_tokens.size(-1)
        _mini_batch['src_tokens'] = src_tokens

        shift_labels = mini_batch['labels'][..., 1:]
        _mini_batch['labels'] = torch.nn.functional.pad(shift_labels, (0, 1), 'constant', IGNORE_IDX).contiguous()

        return {
            "nsentences": len(samples),
            "ntokens": len(samples) * seq_len,
            "net_input": _mini_batch,
            "target": _mini_batch.pop('labels'),
        }

    dataset_config = DatasetConfig(
        type=(lambda split: dataset_disk),
        train_args={'split': 'train'},
    )
    dataloader_config = DataloaderConfig(
        train_args={
            'collate_fn': collate_fn,
            'drop_last': True,
        },
    )
    sampler_config = DatasetSamplerConfig(
        train_args={
            'shuffle': False,
        },
    )
    compute_config = ComputeConfig(
        plan_ngpus=4,
        runtime_ngpus=4,
        constant_folding=True,
        use_zero=True,
        use_end2end=True,
        # autodist config:
        pas_config={
            # 'mem_constraint': 64, # - memory constraint is set to 64GB
            'recompute_modules': 'Phi3DecoderLayer',
        },
    )

    # Load dataset
    print('-' * 50)
    print("Loading dataset...", end=' ')
    stage = 'train'
    dataset_args = getattr(dataset_config, f'{stage}_args')
    if not dataset_args and stage != 'train':
        raise ValueError(f"{stage} dataset will not be created because empty arguments are provided.")

    kwargs = TrainerArgs.create_kwarg(dataset_args)
    dataset_class = load_type(dataset_config.type)
    ta_dataset = dataset_class(**kwargs)
    if isinstance(dataset_class, torch.utils.data.IterableDataset):
        raise ValueError("IterableDataset is not supported")
    print("Done.")

    # Load Sampler
    print('-' * 50)
    print("Loading sampler...", end=' ')
    sampler_args = getattr(sampler_config, f'{stage}_args')
    sampler_args = sampler_args or sampler_config.train_args
    kwargs = TrainerArgs.create_kwarg(sampler_args)
    kwargs['dataset'] = ta_dataset
    kwargs['num_replicas'] = compute_config.runtime_ngpus // compute_config.plan_ngpus
    kwargs['rank'] = int(os.environ.get('RANK', 0)) // compute_config.plan_ngpus
    sampler_class = load_type(sampler_config.type)

    ta_sampler = sampler_class(**kwargs)
    print("Done.")

    # Load DataLoader
    print('-' * 50)
    print("Loading dataloader...", end=' ')
    dataloader_args = getattr(dataloader_config, f'{stage}_args')
    dataloader_args = dataloader_args or dataloader_config.train_args

    kwargs = TrainerArgs.create_kwarg(dataloader_args)
    if 'batch_size' in kwargs:
        raise ValueError("`batch_size` should not be specified in dataloader_args. "
                            "You should use `MICRO_BATCH_SIZE` instead.")

    kwargs['dataset'] = ta_dataset
    if 'collate_fn' in kwargs:
        # special handling for collate_fn as a function
        # here we don't use self.collate_fn to avoid its implementation hacking
        kwargs['collate_fn'] = load_type(kwargs['collate_fn'])
    kwargs['batch_size'] = MICRO_BATCH_SIZE
    kwargs['sampler'] = ta_sampler
    dataloader_class = load_type(dataloader_config.type)

    ta_dataloader = dataloader_class(**kwargs)
    print("Done.")

    trainer_dataloader[stage] = ta_dataloader
    update_freq = GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE // scaling_factor
    data_iter = enumerate(_global_batch_iterator(trainer_dataloader, update_freq, num_skip_first=0, stage='train'))
    return data_iter


def iter_for_val(args, data_iter, model, loss_save_path):
    print('-' * 60)
    print(f"Running validation loop", flush=True)

    num_samples = 0
    losses = []
    for idx, batches in data_iter:
        if idx < GLOBAL_BATCH_OFFSET: continue

        print('-' * 30)
        print(f"Global Batch {idx}", flush=True)
        with torch.no_grad():
            for i, batch in enumerate(batches):
                start_time = time.perf_counter()
                outputs = model(batch)
                infer_time = time.perf_counter() - start_time

                loss, _, _, _ = outputs
                loss_val = loss.cpu().numpy()
                print(f"Batch {idx} | Sample {i} | Loss: {loss_val} | Time: {infer_time:.3f} s", flush=True)
                sys.stdout.flush()

                num_samples += 1
                losses.append(loss_val)
                if args.debug: break
        if args.debug: break
    
    losses = np.array(losses)
    np.save(loss_save_path, losses)

    print('Validation Done.', flush=True)



def print_args(args):
    print('-' * 60)
    print(f"Experiment: {args.expr_name}")
    print(f"GPU Set: {args.gpu_set}")
    print(f"Epoch: {args.epoch_idx}")
    print(f"Iteration: {args.iter_idx}")
    print(f"Use Pretrained Checkpoint: {args.original}")
    print(f"Debug: {args.debug}")
    print('-' * 30)
    print(f"Number of Global Batches: {NUM_GLOBAL_BATCHES}")
    print(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")
    print(f"Micro Batch Size: {MICRO_BATCH_SIZE}")
    print('-' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr_dir", type=str, required=True)
    parser.add_argument("--expr_name", type=str, required=True)
    parser.add_argument("--gpu_set", type=str, required=True)
    parser.add_argument("--epoch_idx", type=int, required=True)
    parser.add_argument("--iter_idx", type=int, required=True)
    parser.add_argument("--original", action="store_true")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument("-o", "--override", action="store_true")
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument('--minfer_config', type=str, default="Phi-3-mini-4k-instruct-LongRoPE-128k")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Setting for Pretrained checkpoint
    if not args.original:
        model_id = f"/scratch/eval/{args.gpu_set}/minfer_phi/{args.expr_name}/checkpoints/{args.epoch_idx:04d}-{args.iter_idx:04d}/merged"
    else:
        print("Using the pretrained(original) model checkpoint...")
        model_id = "microsoft/Phi-3-mini-4k-instruct" if args.model_id is None else args.model_id
        args.epoch_idx = 0
        args.iter_idx = 0

    # ------------------------------------------------------------------
    # Print Settings
    print_args(args)

    # ------------------------------------------------------------------
    # Set log file
    log_file_name = 'validation.log'
    log_file_path = os.path.join(
        f'/scratch/sync/nnscaler_store/{args.gpu_set}', args.expr_dir, args.expr_name,
        'checkpoints', f'{args.epoch_idx:04d}-{args.iter_idx:04d}', 
        log_file_name
    )
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    try:
        log_f = open(log_file_path, 'w')
    except Exception as e:
        print(f"Error opening log file {log_file_path}.")
        log_file_path = log_file_path.replace('/scratch/sync/nnscaler_store', '/scratch/eval')
        log_f = open(log_file_path, 'w')
    
    print(f"Logging to {log_file_path}")
    sys.stdout = log_f
    sys.stderr = sys.stdout
    print_args(args)

    # ------------------------------------------------------------------
    # Merge Checkpoints
    merged = merge_ckpts(args)
    if not merged:
        print(f"Error merging checkpoints. Exiting...")
        exit(1)

    # ------------------------------------------------------------------
    # Copy config files to local directory
    copied = copy_config_files(args)
    if not copied:
        print(f"Error copying config files. Exiting...")
        exit(1)


    # ------------------------------------------------------------------
    # Replace the Attention module 
    print('-' * 60)
    if 'mfmb' not in args.expr_name and 'mf_mb' not in args.expr_name:
        print(f"Using Baseline model for validation...")
        PHI_ATTENTION_CLASSES['flash_attention_2'] = NNScalerPhiFlashAttention2


        model_args = {
            "model_id": model_id,
        }
        if args.original:
            model_args['config_path'] = f"{os.getenv('NNSCALER_HOME')}/experiments/minfer_phi/phi3/lc_config"
        if args.peek_attn_recall:
            model_args['peek_attn_recall'] = True
            model_args['minfer_config_path'] = f"/scratch/nnscaler/experiments/minfer_phi/minfer_modules/configs/{args.minfer_config}.json"

        model = BaselineModel(**model_args)
    else:
        print(f"Using MInfer model for validation...")
        PHI_ATTENTION_CLASSES['flash_attention_2'] = MInferAttention


        if 'mcontrol' in args.expr_name:
            minfer_config_path = os.path.join(MINFER_CONFIG_DIR, f'mfmb_4k_base_iter_10.yaml')
        else:
            minfer_config_path = os.path.join(MINFER_CONFIG_DIR, f'mfmb_4k_config.yaml')
        print(f'Using MInfer config at {minfer_config_path}')

        with open(minfer_config_path, 'r') as f:
            minfer_config = yaml.safe_load(f)
        minfer_config['start_sparse_iter'] = 0
        print(f"MInfer config: {minfer_config}")

        model_args = {
            'model_id': model_id,
            'minfer_config': minfer_config,
        }
        if args.original:
            model_args['config_path'] = f"{os.getenv('NNSCALER_HOME')}/experiments/minfer_phi/phi3/lc_config"
        model = MInferModel(**model_args)
    model.model = model.model.to(DEVICE)

    print('-' * 60)
    print(f"Model Config:")
    print(model.model.config, flush=True)

    # ------------------------------------------------------------------ #
    # Set save path
    sparse_ratio_save_local_dir = os.path.join(
        '/scratch/eval', args.gpu_set, args.expr_dir, args.expr_name, "checkpoints",
        f"{args.epoch_idx:04d}-{args.iter_idx:04d}", "sparse_ratios"
    )
    os.makedirs(sparse_ratio_save_local_dir, exist_ok=True)


    # ------------------------------------------------------------------
    # Get dataloader
    print("Getting dataloader...")
    data_iter = get_dataloader(model_id)
    print("Done.")

    # ------------------------------------------------------------------
    # Start Iteration
    iter_for_val(
        args, data_iter, model, 
        log_file_path.replace('validation.log', 'val_losses.npy')
    )

    if not log_file_path.startswith('/scratch/sync/nnscaler_store'):
        log_f.close()
        print(f"Copying log file to storage account...", end=' ')
        run_res = transfer_by_cp(log_file_path)
        if run_res != 0:
            print(f"Error uploading log file. Exiting...")

    if not args.debug:
        print("Cleaning up...", end=' ')
        run_cmd(["rm", "-rf", os.path.join(
            '/scratch/eval', args.gpu_set, args.expr_dir, args.expr_name, "checkpoints",
            f"{args.epoch_idx:04d}-{args.iter_idx:04d}", "merged",
        )])
        print("Done.")