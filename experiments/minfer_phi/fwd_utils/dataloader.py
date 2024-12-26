import os
import torch

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from typing import List, Optional, Tuple, Union, Any, Dict

from nnscaler.utils import set_default_logger_level
from nnscaler.cli.trainer_args import (
    DatasetConfig,
    TrainerArgs,
    DataloaderConfig,
    DatasetSamplerConfig,
    load_type
)

# how to import `get_tokenizer` in /scratch/nnscaler/experiments/minfer_phi/train.py?
import sys
sys.path.append('/scratch/nnscaler/experiments/minfer_phi')
from train import get_tokenizer


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

