import os
import time
import math
import yaml
import torch
import datetime
import datasets
import argparse
import builtins
import importlib
import subprocess
import numpy as np
import huggingface_hub

from datasets import load_from_disk
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union, Any, Dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    
)

from flash_attn import flash_attn_func, flash_attn_varlen_func


from nnscaler.utils import set_default_logger_level
from nnscaler.cli.trainer_args import (
    CheckpointConfig,
    DatasetConfig,
    HookMapConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerArgs,
    DataloaderConfig,
    AggregatedOutputs,
    LogConfig,
    DatasetSamplerConfig,
    load_type
)
from nnscaler.parallel import ComputeConfig, BroadcastGenFilesStrategy
from nnscaler.runtime.f16_optimizer import MixedPrecisionAdamW
from nnscaler.cli.loggers.tensorboard import TensorBoardLogger

from custom_trainer import get_iter_cnt, need_save_data
from phi3 import (
    Phi3Config as PhiConfig, Phi3Attention as PhiAttention, 
    PHI3_ATTENTION_CLASSES as PHI_ATTENTION_CLASSES, 
    apply_rotary_pos_emb, repeat_kv, _get_unpad_data
)
from modeling_modifier import NNScalerPhiFlashAttention2
from minfer_modifier_v2 import MInferAttention
from train import get_tokenizer, BaselineModel, MInferModel


with open("/blob/.secrets/sas_token", "r") as f:
    SAS_TOKEN = f.read().strip()
SAS_POSTFIX = f"?{SAS_TOKEN}"

IGNORE_IDX = -100
DEVICE_1 = 'cuda:1'
DEVICE_2 = 'cuda:2'

SEQ_LEN = 131072
LAST_Q = 512
TOPK = 1024

NUM_GLOBAL_BATCHES = 2
GLOBAL_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 1

# NUM_GLOBAL_BATCHES = 2
# GLOBAL_BATCH_SIZE = 5
# MICRO_BATCH_SIZE = 1

NUM_LAYERS = 32
NUM_HEADS = 32

def run_cmd(cmd):
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error running command {cmd}. Exiting...")
        return 1
    return 0

def transfer_by_cp(local_path, dir='upload'):
    remote_path = local_path.replace('/scratch/eval', '/blob/nnscaler_store')
    os.makedirs(os.path.dirname(remote_path), exist_ok=True)
    if dir == 'upload':
        run_res = run_cmd([
            'cp', local_path, remote_path
        ])
    else:
        run_res = run_cmd([
            'cp', remote_path, local_path
        ])
    
    if run_res != 0:
        print(f"Error {'uploading' if dir == 'upload' else 'downloading'} {local_path}. Exiting...")
        return 1
    return 0


def _fix_input(input, input_dtype):
    if isinstance(input, dict):
        return {k: _fix_input(v, input_dtype) for k, v in input.items()}
    elif isinstance(input, list):
        return [_fix_input(v, input_dtype) for v in input]
    elif isinstance(input, tuple):
        return tuple(_fix_input(v, input_dtype) for v in input)
    elif isinstance(input, torch.Tensor):
        if input.is_floating_point() and input_dtype is not None:
            return input.to(input_dtype).to(DEVICE_1)
        else:
            return input.to(DEVICE_1)
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


class BaselineAttentionWSparse(NNScalerPhiFlashAttention2):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.sparse_ratio: torch.Tensor = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        res = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ---------------------------------------
        # print('-' * 30)
        causal_mask = torch.arange(q_len - LAST_Q, q_len, device=DEVICE_2)[:, None] >= torch.arange(q_len, device=DEVICE_2)[None, :]
        
        # if self.layer_idx == 0:
        #     print('-' * 60)
        #     print(f"Before QK^T: query_states:\n{query_states}\nkey_states:\n{key_states}\n")
        qk = torch.einsum(
            f'bhmk, bhnk -> bhmn', 
            query_states[:, :, -LAST_Q:, :].contiguous().to(DEVICE_2), # [BATCH, N_HEADS, LAST_Q, D_HEAD]
            key_states.to(DEVICE_2), # [BATCH, N_HEADS, N_CTX, D_HEAD]
        ) / math.sqrt(self.head_dim) # [BATCH, N_HEADS, LAST_Q, N_CTX]

        qk = torch.where(causal_mask, qk, float('-inf'))
        del causal_mask


        attn_weights = torch.nn.functional.softmax(qk, dim=-1)

        # Calculate the sum of topk elements at each row of the attention weights
        topk_values, _ = torch.topk(attn_weights, TOPK, dim=-1)
        self.sparse_ratio = torch.sum(topk_values, dim=-1).cpu() # [BATCH, N_HEAD, LAST_Q]
        del qk, topk_values, attn_weights
        
        return res

class MInferAttentionWSparse(MInferAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        res = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ---------------------------------------
        # print('-' * 30)
        causal_mask = torch.arange(q_len - LAST_Q, q_len, device=DEVICE_2)[:, None] >= torch.arange(q_len, device=DEVICE_2)[None, :]
        
        # if self.layer_idx == 0:
        #     print('-' * 60)
        #     print(f"Before QK^T: query_states:\n{query_states}\nkey_states:\n{key_states}\n")
        qk = torch.einsum(
            f'bhmk, bhnk -> bhmn', 
            query_states[:, :, -LAST_Q:, :].contiguous().to(DEVICE_2), # [BATCH, N_HEADS, LAST_Q, D_HEAD]
            key_states.to(DEVICE_2), # [BATCH, N_HEADS, N_CTX, D_HEAD]
        ) / math.sqrt(self.head_dim) # [BATCH, N_HEADS, LAST_Q, N_CTX]
        qk = torch.where(causal_mask, qk, float('-inf'))
        del causal_mask


        attn_weights = torch.nn.functional.softmax(qk, dim=-1)

        # Calculate the sum of topk elements at each row of the attention weights
        topk_values, _ = torch.topk(attn_weights, TOPK, dim=-1)
        self.sparse_ratio = torch.sum(topk_values, dim=-1).cpu() # [BATCH, N_HEAD, LAST_Q]
        del qk, topk_values, attn_weights

        return res

class BaselineSparseModel(BaselineModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.model = self.model.to(DEVICE_1)
        self.model.eval()

    def get_sparse_ratio(self):
        sparse_ratios = []
        for layer in self.model.model.layers:
            sparse_ratio = layer.self_attn.sparse_ratio
            batch_size, q_len, _ = sparse_ratio.size()
            sparse_ratios.append(sparse_ratio.view(batch_size, q_len, LAST_Q))

        sparse_ratios = torch.stack(sparse_ratios, dim=1)
        return sparse_ratios

class MInferSparseModel(MInferModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.model = self.model.to(DEVICE_1)
        self.model.eval()

    def get_sparse_ratio(self):
        sparse_ratios = []
        for layer in self.model.model.layers:
            sparse_ratio = layer.self_attn.sparse_ratio
            batch_size, q_len, _ = sparse_ratio.size()
            sparse_ratios.append(sparse_ratio.view(batch_size, q_len, LAST_Q))

        sparse_ratios = torch.stack(sparse_ratios, dim=1)
        return sparse_ratios


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
    parser.add_argument("--model_id", type=str, default=None)
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
    print(f"Last Q: {LAST_Q}")
    print(f"TopK: {TOPK}")
    print('-' * 60)

    # ------------------------------------------------------------------
    # Set log file
    import sys
    log_file_path = os.path.join(
        f'/blob/nnscaler_store/{args.gpu_set}', args.expr_dir, args.expr_name,
        'checkpoints', f'{args.epoch_idx:04d}-{args.iter_idx:04d}', 'sparse_ratio.log'
    )
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    try:
        log_f = open(log_file_path, 'w')
    except Exception as e:
        print(f"Error opening log file {log_file_path}.")
        log_file_path = log_file_path.replace('/blob/nnscaler_store', '/scratch/eval')
        log_f = open(log_file_path, 'w')

    print(f"Logging to {log_file_path}")
    sys.stdout = log_f
    sys.stderr = sys.stdout

    print(f"Current time: {datetime.datetime.now()}")
    print('-' * 60)
    print(f"Experiment: {args.expr_name}")
    print(f"Model ID: {model_id}")
    print(f"GPU Set: {args.gpu_set}")
    print(f"Epoch: {args.epoch_idx}")
    print(f"Iteration: {args.iter_idx}")
    print(f"Use Pretrained Checkpoint: {args.original}")
    print(f"Debug: {args.debug}")
    print('-' * 30)
    print(f"Number of Global Batches: {NUM_GLOBAL_BATCHES}")
    print(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")
    print(f"Micro Batch Size: {MICRO_BATCH_SIZE}")
    print(f"Last Q: {LAST_Q}")
    print(f"TopK: {TOPK}")
    print('-' * 60)


    # ------------------------------------------------------------------
    # Replace the Attention module 
    if 'mfmb' not in args.expr_name and 'mf_mb' not in args.expr_name:
        print(f"Using Baseline model for {args.expr_name}...")
        PHI_ATTENTION_CLASSES['flash_attention_2'] = BaselineAttentionWSparse
        model_args = {
            "model_id": model_id,
        }
        if args.original:
            model_args['config_path'] = f"{os.getenv('NNSCALER_HOME')}/experiments/minfer_phi/phi3/lc_config"
        model = BaselineSparseModel(**model_args)
    else:
        print(f"Using MInfer model for {args.expr_name}...")
        PHI_ATTENTION_CLASSES['flash_attention_2'] = MInferAttentionWSparse

        model_args = {
            "model_id": model_id,
            "minfer_config": {
                # MInfer's config is the one searched from the pretrained (4k) version with LongRoPE
                'config_path': "/scratch/nnscaler/experiments/minfer_phi/minfer_modules/configs/Phi-3-mini-4k-instruct-LongRoPE-128k.json",
            },
        }
        if args.original:
            model_args['config_path'] = f"{os.getenv('NNSCALER_HOME')}/experiments/minfer_phi/phi3/lc_config"
        model = MInferSparseModel(**model_args)
    print('-' * 60)
    print(f"Model Config:")
    print(model.model.config)
    
    # ------------------------------------------------------------------#
    # Set save path
    sparse_ratio_save_url = (
        f"https://chengzhang.blob.core.windows.net/wenxuanli/nnscaler_store/"
        f"{args.gpu_set}/{args.expr_dir}/{args.expr_name}/checkpoints/"
        f"{args.epoch_idx:04d}-{args.iter_idx:04d}/sparse_ratios/"
    )
    sparse_ratio_save_local_dir = os.path.join(
        '/scratch/eval', args.gpu_set, args.expr_dir, args.expr_name, "checkpoints",
        f"{args.epoch_idx:04d}-{args.iter_idx:04d}", "sparse_ratios"
    )
    os.makedirs(sparse_ratio_save_local_dir, exist_ok=True)


    # ------------------------------------------------------------------
    # Get dataloader
    print("Getting dataloader...", end=" ")
    data_iter = get_dataloader(model_id)
    print("Done.")

    # ------------------------------------------------------------------
    # Iterate over the batches within a global batch
    sys.stdout.flush()
    for idx, batches in data_iter:
        if idx >= NUM_GLOBAL_BATCHES: break
        print('-' * 60)
        print(f"Global batch {idx} with {len(batches)} micro batches")
        save_url_idx = sparse_ratio_save_url + f"{str(idx)}.npy"
        save_url_loss_idx = sparse_ratio_save_url + f"{str(idx)}_loss.npy"
        save_local_idx = os.path.join(sparse_ratio_save_local_dir, f"{str(idx)}.npy")
        save_local_loss_idx = os.path.join(sparse_ratio_save_local_dir, f"{str(idx)}_loss.npy")
        if (os.path.exists(save_local_idx) 
            or os.path.exists(save_local_idx.replace('/scratch/eval', '/blob/nnscaler_store'))) \
            and not args.override:
            print(f"Skipping batch {idx} because {save_local_idx} exists (use '--override' to override).")
            sys.stdout.flush()
            continue

        sparse_ratios = np.zeros((GLOBAL_BATCH_SIZE, NUM_LAYERS, NUM_HEADS, LAST_Q))
        losses = np.zeros((GLOBAL_BATCH_SIZE, ))
        with torch.no_grad():
            for i, batch in enumerate(batches):
                print("-" * 30)
                start_time = time.perf_counter()
                outputs = model(batch)
                infer_time = time.perf_counter() - start_time

                loss, _, _, _ = outputs

                micro_sparse_ratios = model.get_sparse_ratio() # [1, LAYER, HEAD, LAST_Q]

                sparse_ratios[i] = micro_sparse_ratios.squeeze(0).cpu().numpy()
                losses[i] = loss.cpu().numpy()
                print(f"Batch {idx} | Sample {i} | Loss: {losses[i]} | Time: {infer_time:.3f} s")
                sys.stdout.flush()

        print(f'Saving sparse ratio for batch {idx}...', end=' ')
        
        np.save(save_local_idx, sparse_ratios)
        print('Done.')

        print(f'Saving loss for batch {idx}...', end=' ')
        np.save(save_local_loss_idx, losses)
        print('Done.')

        print(f'Uploading sparse ratio for batch {idx}...', end=' ')
        run_res = transfer_by_cp(save_local_idx)
        if run_res != 0:
            print(f"Error uploading sparse ratio for batch {idx}. Exiting...")
        print('Done.')

        print(f'Uploading loss for batch {idx}...', end=' ')
        run_res = transfer_by_cp(save_local_loss_idx)
        if run_res != 0:
            print(f"Error uploading loss for batch {idx}. Exiting...")
        print('Done.')

        if args.debug: break
    
    if not log_file_path.startswith('/blob/nnscaler_store'):
        log_f.close()
        print(f"Copying log file to blob storage...", end=' ')
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