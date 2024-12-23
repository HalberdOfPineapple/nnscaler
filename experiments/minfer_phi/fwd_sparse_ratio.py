import os
import json
import time
import math
import torch
import argparse
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

with open("/scratch/sync/.secrets/sas_token", "r") as f:
    SAS_TOKEN = f.read().strip()
SAS_POSTFIX = f"?{SAS_TOKEN}"

IGNORE_IDX = -100
DEVICE_1 = 'cuda:1'
DEVICE_2 = 'cuda:2'

SEQ_LEN = 131072
LAST_Q = 512
K_BLOCK_SIZE = 4096
TOPK = 1024
NUM_ATTN_BLOCKS = 10
NUM_ATTN_SAMPLES = 10

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
    remote_path = local_path.replace('/scratch/eval', '/scratch/sync/nnscaler_store')
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


def cal_last_QK(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    past_key_value: Optional[Cache],
    return_qkv: bool = False,
):
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
    if return_qkv:
        return attn_weights, query_states, key_states, value_states
    else:
        return attn_weights, None, None, None


def build_sparse_mask(
    block_count: torch.Tensor, # [BATCH, N_HEADS, NUM_ROWS], note that NUM_ROWS means the number of 64-sized rows
    block_offset, # [BATCH, N_HEADS, NUM_ROWS, NNZ_S], which refers to the start of the non-sparse K/V blocks to be computed with the corresponding Q block
    column_count, # [BATCH, N_HEADS, NUM_ROWS]
    column_index, # [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    attn_mask = torch.zeros((MICRO_BATCH_SIZE, LAST_Q, SEQ_LEN), device=DEVICE_2)
    row_offset = (SEQ_LEN - LAST_Q) // block_size_M
    num_row_blocks = LAST_Q // block_size_M

    for batch_idx in range(MICRO_BATCH_SIZE):
        for row_idx in range(row_offset, row_offset + num_row_blocks):
            block_cnt = block_count[batch_idx, 0, row_idx]
            block_off = block_offset[batch_idx, 0, row_idx]
            col_cnt = column_count[batch_idx, 0, row_idx]
            col_idx = column_index[batch_idx, 0, row_idx]

            row_start = row_idx * block_size_M - (SEQ_LEN - LAST_Q)
            row_end = min(LAST_Q, row_start + block_size_M)
            print(f"Row-Dimension Start: {row_start} End: {row_end}")

            for i in range(block_cnt):
                curr_block_start = block_off[i]
                curr_block_end = min(SEQ_LEN, curr_block_start + block_size_N)
                print(f"Column-Dimension Start: {curr_block_start} End: {curr_block_end}")

                # attn_mask[batch_idx, row_start:row_end, curr_block_start:curr_block_end] = 1
                attn_mask[batch_idx, row_start:row_end, curr_block_start:curr_block_end] = \
                    torch.ones((row_end - row_start, curr_block_end - curr_block_start), device=DEVICE_2)

            for j in range(col_cnt):
                col_index = col_idx[j]
                # attn_mask[batch_idx, row_start:row_end, col_index] = 1
                attn_mask[batch_idx, row_start:row_end, col_index] = \
                    torch.ones((row_end - row_start), device=DEVICE_2)

    return attn_mask

def get_sparse_ratio(
    attn_weights: torch.Tensor,
    topk: int = TOPK,
):
    topk_values, _ = torch.topk(attn_weights, topk, dim=-1)
    sparse_ratio = torch.sum(topk_values, dim=-1).cpu()
    return sparse_ratio

def save_attn_blocks(
    attn_weights: torch.Tensor,
    k_bsz: int = K_BLOCK_SIZE,
):
    # attn_weights - [BATCH, N_HEADS, LAST_Q, SEQ_LEN]
    attn_blocks = []
    starting_indices = list(range(0, SEQ_LEN - K_BLOCK_SIZE, (SEQ_LEN - K_BLOCK_SIZE) // (NUM_ATTN_BLOCKS - 1) ))[:-1]
    starting_indices.append(SEQ_LEN - K_BLOCK_SIZE)
    for i in starting_indices:
        attn_block = attn_weights[..., i:i+k_bsz].cpu().numpy() # [BATCH, N_HEADS, LAST_Q, K_BLOCK_SIZE]
        attn_blocks.append(attn_block)

    attn_blocks = np.stack(attn_blocks, axis=2) # [BATCH, N_HEADS, N_BLOCKS, LAST_Q, K_BLOCK_SIZE]
    return attn_blocks

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
        self.attn_blocks: np.ndarray = None

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

        attn_weights, query_states, key_states, value_states = \
                self.cal_last_QK(
                    hidden_states, position_ids, past_key_value, 
                    return_qkv=self.peek_attn_recall,
                )
        self.sparse_ratio = get_sparse_ratio(attn_weights)
        
        if self.save_attn:
            self.attn_blocks = save_attn_blocks(attn_weights, k_bsz=K_BLOCK_SIZE) # [BATCH, N_HEADS, N_BLOCKS, LAST_Q, K_BLOCK_SIZE]
            # print(f"BaselineAttentionWSparse attn_blocks shape: {self.attn_blocks.shape}")
        
        if self.peek_attn_recall:
            attn_mask = []
            for head_idx in range(NUM_HEADS):
                print(f"Peeking Attention Recall for Layer {self.layer_idx}, Head {head_idx}...", flush=True)
                pattern = self.best_pattern.get(head_idx, ("vertical_and_slash", 100, 6096, 1))
                ty, vertical_size, slash_size, _ = pattern
                print(f"Layer {self.layer_idx} Head {head_idx} Pattern: {pattern}", flush=True)

                block_count, block_offset, column_count, column_index, _ = gen_block_indices(
                    query_states[:, head_idx].unsqueeze(1) , 
                    key_states[:, head_idx].unsqueeze(1) ,
                    value_states[:, head_idx].unsqueeze(1) ,
                    SEQ_LEN, vertical_size, slash_size, self.head_dim,
                    block_size_M=64, block_size_N=64
                )

                head_mask = build_sparse_mask(block_count, block_offset, column_count, column_index)
                print(f"Layer {self.layer_idx} Head {head_idx} head_mask: {head_mask}", flush=True)
                print(f"Layer {self.layer_idx} Head {head_idx} is all zero: {torch.all(head_mask == 0)}", flush=True)

                attn_mask.append(head_mask)
            attn_mask = torch.stack(attn_mask, dim=1) # [BATCH, N_HEADS, LAST_Q, SEQ_LEN]
            # convert attn_mask to bool
            attn_mask = attn_mask.bool()
            print(f"Layer {self.layer_idx} attn_mask: {attn_mask}", flush=True)

            causal_mask = torch.arange(SEQ_LEN - LAST_Q, SEQ_LEN, device=DEVICE_2)[:, None] >= torch.arange(SEQ_LEN, device=DEVICE_2)[None, :]

            qk = torch.einsum(
                f'bhmk, bhnk -> bhmn', 
                query_states[:, :, -LAST_Q:, :].contiguous().to(DEVICE_2), # [BATCH, N_HEADS, LAST_Q, D_HEAD]
                key_states.to(DEVICE_2), # [BATCH, N_HEADS, N_CTX, D_HEAD]
            ) / math.sqrt(self.head_dim) # [BATCH, N_HEADS, LAST_Q, N_CTX]

            print(f"Layer {self.layer_idx} QK: {qk} before masking", flush=True)
            qk = torch.where(attn_mask & causal_mask[None, None, :, :], qk, float('-inf'))
            print(f"Layer {self.layer_idx} QK: {qk} after masking", flush=True)
            
            sparse_attn_weights = torch.nn.functional.softmax(qk, dim=-1)
            self.attn_recalls = torch.sum(sparse_attn_weights, dim=-1).cpu().numpy() # [BATCH, N_HEADS, LAST_Q]

        return res
    
    def init_attn_recall(self, minfer_config_path: str):
        self.best_pattern = {int(ii): jj for ii, jj in json.load(open(minfer_config_path))[self.layer_idx].items()}
        print(f"[BaselineAttentionWSparse] MInfer config for {len(self.best_pattern)} heads in Layer {self.layer_idx} initialized")

        
class MInferAttentionWSparse(MInferAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.sparse_ratio: torch.Tensor = None
        self.attn_blocks: np.ndarray = None

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

        attn_weights = self.cal_last_QK(hidden_states, position_ids, past_key_value)
        self.sparse_ratio = get_sparse_ratio(attn_weights)

        if self.save_attn:
            # np.array
            # [BATCH, N_HEADS, N_BLOCKS, LAST_Q, K_BLOCK_SIZE]
            self.attn_blocks = save_attn_blocks(attn_weights, k_bsz=K_BLOCK_SIZE)
            # print(f"MInferAttentionWSparse attn_blocks shape: {self.attn_blocks.shape}")
        
        return res

class BaselineSparseModel(BaselineModel):
    def __init__(
            self, 
            save_attn: bool=False, 
            peek_attn_recall: bool=False,
            minfer_config_path: str = None,
            **kwargs
        ):
        super().__init__(**kwargs)

        if peek_attn_recall and minfer_config_path is None:
            raise ValueError("minfer_config_path is required for attention recall peeking.")
        
        self.model = self.model.to(DEVICE_1)
        self.model.eval()

        Attention = self.model.model.layers[0].self_attn.__class__
        def update_module(m):
            if isinstance(m, Attention):
                m.cal_last_QK = cal_last_QK.__get__(
                    m, Attention
                )
                m.save_attn = save_attn

                m.peek_attn_recall = peek_attn_recall
                if peek_attn_recall:
                    m.init_attn_recall(minfer_config_path)

        self.model.apply(update_module)

    def get_sparse_ratio(self):
        sparse_ratios = []
        for layer in self.model.model.layers:
            sparse_ratio = layer.self_attn.sparse_ratio
            batch_size, q_len, _ = sparse_ratio.size()
            sparse_ratios.append(sparse_ratio.view(batch_size, q_len, LAST_Q))

        sparse_ratios = torch.stack(sparse_ratios, dim=1)
        return sparse_ratios
    
    def get_attn_blocks(self):
        attn_blocks = []
        for layer in self.model.model.layers:
            attn_block = layer.self_attn.attn_blocks # [BATCH, N_HEADS, N_BLOCKS, LAST_Q, K_BLOCK_SIZE]
            attn_blocks.append(attn_block)

        attn_blocks = np.stack(attn_blocks, axis=1) # [BATCH, NUM_LAYERS, N_HEADS, N_BLOCKS, LAST_Q, K_BLOCK_SIZE]
        return attn_blocks
    
    def get_attn_recalls(self):
        attn_recalls = []
        for i, layer in enumerate(self.model.model.layers):
            try:
                attn_recall = layer.self_attn.attn_recalls # [BATCH, N_HEADS, LAST_Q]
                attn_recalls.append(attn_recall)
            except AttributeError:
                print(f"[Warning] Layer {i} does not have attn_recalls attribute. Skipping...")
                continue

        attn_recalls = np.stack(attn_recalls, axis=1) # [BATCH, NUM_LAYERS, N_HEADS, LAST_Q]
        return attn_recalls

class MInferSparseModel(MInferModel):
    def __init__(self, save_attn: bool=False, **kwargs):
        super().__init__(**kwargs)
        
        self.model = self.model.to(DEVICE_1)
        self.model.eval()

        Attention = self.model.model.layers[0].self_attn.__class__
        def update_module(m):
            if isinstance(m, Attention):
                m.cal_last_QK = cal_last_QK.__get__(
                    m, Attention
                )
                m.save_attn = save_attn

        self.model.apply(update_module)

    def get_sparse_ratio(self):
        sparse_ratios = []
        for layer in self.model.model.layers:
            sparse_ratio = layer.self_attn.sparse_ratio
            batch_size, q_len, _ = sparse_ratio.size()
            sparse_ratios.append(sparse_ratio.view(batch_size, q_len, LAST_Q))

        sparse_ratios = torch.stack(sparse_ratios, dim=1)
        return sparse_ratios
    
    def get_attn_blocks(self):
        attn_blocks = []
        for layer in self.model.model.layers:
            attn_block = layer.self_attn.attn_blocks # [BATCH, N_HEADS, N_BLOCKS, LAST_Q, K_BLOCK_SIZE]
            attn_blocks.append(attn_block)

        attn_blocks = np.stack(attn_blocks, axis=1) # [BATCH, NUM_LAYERS, N_HEADS, N_BLOCKS, LAST_Q, K_BLOCK_SIZE]
        return attn_blocks

def iter_for_sparse_ratio(args):
    print('-' * 60)
    print(f"Running Sparse Ratio Measurement Mode")
    
    sys.stdout.flush()
    for idx, batches in data_iter:
        if idx >= NUM_GLOBAL_BATCHES: break
        print('-' * 60)
        print(f"Global batch {idx} with {len(batches)} micro batches")
        save_local_idx = os.path.join(sparse_ratio_save_local_dir, f"{str(idx)}.npy")
        save_local_loss_idx = os.path.join(sparse_ratio_save_local_dir, f"{str(idx)}_loss.npy")
        if (os.path.exists(save_local_idx) 
            or os.path.exists(save_local_idx.replace('/scratch/eval', '/scratch/sync/nnscaler_store'))) \
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


def iter_for_attn_peek(args):
    print('-' * 60)
    print(f"Running Peek Attention Mode", flush=True)
    global NUM_ATTN_BLOCKS, NUM_ATTN_SAMPLES
    if args.num_peek_blocks != NUM_ATTN_BLOCKS:
        NUM_ATTN_BLOCKS = args.num_peek_blocks
    if args.num_peek_samples != NUM_ATTN_SAMPLES:
        NUM_ATTN_SAMPLES = args.num_peek_samples

    
    attn_block_save_dir = os.path.join(
        '/scratch/sync', "nnscaler_store", 
        args.gpu_set, args.expr_dir, args.expr_name, "checkpoints",
        f"{args.epoch_idx:04d}-{args.iter_idx:04d}", "attn_blocks"
    )
    local_attn_save_dir = attn_block_save_dir.replace('/scratch/sync', '/scratch/eval')
    os.makedirs(attn_block_save_dir, exist_ok=True)
    os.makedirs(local_attn_save_dir, exist_ok=True)

    num_samples = 0
    for idx, batches in data_iter:
        attn_blocks = np.zeros((NUM_LAYERS, NUM_HEADS, NUM_ATTN_BLOCKS, LAST_Q, K_BLOCK_SIZE))
        with torch.no_grad():
            for i, batch in enumerate(batches):
                print("-" * 30)
                start_time = time.perf_counter()
                outputs = model(batch)
                infer_time = time.perf_counter() - start_time

                loss, _, _, _ = outputs

                attn_blocks = model.get_attn_blocks()[0] # [NUM_LAYERS, N_HEADS, N_BLOCKS, LAST_Q, K_BLOCK_SIZE]
                print(f"Batch {idx} | Sample {i} | Loss: {loss.cpu().numpy()} | Time: {infer_time:.3f} s", flush=True)

                print(f'Saving sparse ratio for batch {idx}...')
                for l in range(NUM_LAYERS):
                    if args.expr_name == "phi_lc_4k_131072" and l <= 2: continue
                    for h in range(NUM_HEADS):
                        if args.expr_name == "phi_lc_4k_131072" and l == 3 and h <= 15: continue
                        for block_idx in range(NUM_ATTN_BLOCKS):
                            print(f"Saving attn block for layer {l}, head {h}, block {block_idx}...", end=' ', flush=True)
                            attn_block_save_path = os.path.join(
                                attn_block_save_dir,
                                f"sample_{num_samples}",
                                f"layer_{l}", f"head_{h}", f"block_{block_idx}.npy"
                            )
                            os.makedirs(os.path.dirname(attn_block_save_path), exist_ok=True)
                            np.save(attn_block_save_path, attn_blocks[l][h][block_idx])
                            print('Done.', flush=True)

                print('Done.')
                sys.stdout.flush()

                num_samples += 1
                if num_samples >= NUM_ATTN_SAMPLES: break
                if args.debug: break
        if num_samples >= NUM_ATTN_SAMPLES: break


def iter_for_attn_recall(args):
    print('-' * 60)
    print(f"Running Attention Recall Peeking", flush=True)
    global NUM_ATTN_BLOCKS, NUM_ATTN_SAMPLES
    if args.num_peek_blocks != NUM_ATTN_BLOCKS:
        NUM_ATTN_BLOCKS = args.num_peek_blocks
    if args.num_peek_samples != NUM_ATTN_SAMPLES:
        NUM_ATTN_SAMPLES = args.num_peek_samples

    
    attn_recall_save_dir = os.path.join(
        '/scratch/sync', "nnscaler_store", 
        args.gpu_set, args.expr_dir, args.expr_name, "checkpoints",
        f"{args.epoch_idx:04d}-{args.iter_idx:04d}", "attn_recalls"
    )
    local_attn_save_dir = attn_recall_save_dir.replace('/scratch/sync', '/scratch/eval')
    os.makedirs(attn_recall_save_dir, exist_ok=True)
    os.makedirs(local_attn_save_dir, exist_ok=True)

    num_samples = 0
    for idx, batches in data_iter:
        with torch.no_grad():
            for i, batch in enumerate(batches):
                print("-" * 30)
                start_time = time.perf_counter()
                outputs = model(batch)
                infer_time = time.perf_counter() - start_time

                loss, _, _, _ = outputs
                print(f"Batch {idx} | Sample {i} | Loss: {loss.cpu().numpy()} | Time: {infer_time:.3f} s", flush=True)

                print(f'Saving attn recall for batch {idx}...', end=' ', flush=True)
                attn_recalls = model.get_attn_recalls()[0] # [NUM_LAYERS, N_HEADS, LAST_Q]
                attn_recall_save_path = os.path.join(
                    attn_recall_save_dir,
                    f"sample_{num_samples}.npy"
                )
                os.makedirs(os.path.dirname(attn_recall_save_path), exist_ok=True)
                np.save(attn_recall_save_path, attn_recalls)
                print('Done.', flush=True)

                num_samples += 1
                if num_samples >= NUM_ATTN_SAMPLES: break
                if args.debug: break
        if num_samples >= NUM_ATTN_SAMPLES: break
        if args.debug: break

def print_args(args):
    print('-' * 60)
    print(f"Mode: {'Sparse Ratio Measurement' if not args.peek_attn else 'Peek Attention'}")
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
    if args.peek_attn:
        print(f"Number of Peek Blocks: {args.num_peek_blocks}")
        print(f"Number of Peek Samples: {args.num_peek_samples}")
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
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument('--minfer_config', type=str, default="Phi-3-mini-4k-instruct-LongRoPE-128k")
    parser.add_argument('-p', "--peek_attn", action='store_true')
    parser.add_argument('--num_peek_blocks', type=int, default=10)
    parser.add_argument('--num_peek_samples', type=int, default=10)
    parser.add_argument('--peek_attn_recall', action='store_true')
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
    import sys
    if args.peek_attn:
        log_file_name = 'attn_peek.log'
    elif args.peek_attn_recall:
        log_file_name = 'attn_recall_peek.log'
    else:
        log_file_name = 'sparse_ratio.log'

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
    # Replace the Attention module 
    if 'mfmb' not in args.expr_name and 'mf_mb' not in args.expr_name:
        print(f"Using Baseline model for {args.expr_name}...")
        if args.peek_attn_recall:
            print(f"Attention Recall Peeking is enabled for baseline model.")

        PHI_ATTENTION_CLASSES['flash_attention_2'] = BaselineAttentionWSparse
        model_args = {
            "model_id": model_id,
        }
        if args.original:
            model_args['config_path'] = f"{os.getenv('NNSCALER_HOME')}/experiments/minfer_phi/phi3/lc_config"
        if args.peek_attn_recall:
            model_args['peek_attn_recall'] = True
            model_args['minfer_config_path'] = f"/scratch/nnscaler/experiments/minfer_phi/minfer_modules/configs/{args.minfer_config}.json"

        model = BaselineSparseModel(save_attn=args.peek_attn, **model_args)
        
    else:
        print(f"Using MInfer model for {args.expr_name}...")
        if args.peek_attn_recall:
            raise ValueError("Attention Recall Peeking is not supported for MInfer model.")

        PHI_ATTENTION_CLASSES['flash_attention_2'] = MInferAttentionWSparse

        minfer_config_path = f"/scratch/nnscaler/experiments/minfer_phi/minfer_modules/configs/{args.minfer_config}.json"
        print(f"Using MInfer config at {minfer_config_path}")

        model_args = {
            "model_id": model_id,
            "minfer_config": {
                'config_path': minfer_config_path,
            },
        }
        if args.original:
            model_args['config_path'] = f"{os.getenv('NNSCALER_HOME')}/experiments/minfer_phi/phi3/lc_config"
        model = MInferSparseModel(save_attn=args.peek_attn, **model_args)

    print('-' * 60)
    print(f"Model Config:")
    print(model.model.config)
    
    # ------------------------------------------------------------------#
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
    if args.peek_attn:
        iter_for_attn_peek(args)
    elif args.peek_attn_recall:
        iter_for_attn_recall(args)
    else:
        iter_for_sparse_ratio(args)
    
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