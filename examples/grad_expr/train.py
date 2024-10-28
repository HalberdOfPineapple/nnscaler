#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
import os

import datasets
from datasets import load_from_disk
import huggingface_hub
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from modeling_modifier import nnscaler_llama_init
from chunk_linear_cross_entropy import chunk_linear_cross_entropy
from custom_trainer import CustomTrainer as Trainer # from nnscaler.cli.trainer import Trainer

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
)
from nnscaler.parallel import ComputeConfig, BroadcastGenFilesStrategy
from nnscaler.runtime.f16_optimizer import MixedPrecisionAdamW
from nnscaler.cli.loggers.tensorboard import TensorBoardLogger

import logging
logger = logging.getLogger(__name__)

IGNORE_IDX = -100


def get_tokenizer(tokenizer_name_or_path,
                  model_max_length=None,
                  default_bos_token="<s>",
                  default_eos_token="</s>",
                  default_pad_token="[PAD]",
                  default_unk_token="<unk>"):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = default_pad_token
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = default_eos_token
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = default_bos_token
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = default_unk_token

    tokenizer.add_special_tokens(special_tokens_dict)
    if model_max_length:
        tokenizer.model_max_length = model_max_length
    return tokenizer


class WrapperModel(torch.nn.Module):
    def __init__(self, model_id, attn_type: str='custom'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            attn_implementation='flash_attention_2' if attn_type != 'custom' else 'eager'
        )
        self.attn_type = attn_type
        self.iter_idx = 0

    def forward(self, samples):
        outputs = self.model.model(
            input_ids=samples['net_input']['src_tokens'],
            use_cache=False,
            return_dict=False,
        )
        hidden_states = outputs[0]
        losses = chunk_linear_cross_entropy(hidden_states, self.model.lm_head.weight, samples['target'], IGNORE_IDX, 1024)
        loss = torch.sum(losses)

        return loss, loss.data, samples['ntokens'], samples['nsentences']


def aggregate_outputs_fn(loss_outputs, sync_group) -> AggregatedOutputs:
    losses, ntokens_info = [], []
    for _, loss, ntokens, _ in loss_outputs:
        losses.append(loss)
        ntokens_info.append(ntokens)

    loss_sum = torch.sum(torch.stack(losses), dtype=torch.float64)
    torch.distributed.all_reduce(loss_sum, group=sync_group)

    ntokens_sum = torch.sum(torch.tensor(ntokens_info, dtype=torch.float64, device=torch.cuda.current_device()))
    torch.distributed.all_reduce(ntokens_sum, group=sync_group)
    
    num_batches = torch.tensor(len(losses), device=torch.cuda.current_device())
    torch.distributed.all_reduce(num_batches, group=sync_group)

    return AggregatedOutputs(
        loss_sum=loss_sum.item() / ntokens_sum.item(),
        num_batches=num_batches.item(),
        num_tokens=ntokens_sum.item(),
    )


def main(args):

    if args.run_mode == 'run':
        broadcast_strategy = 'all'
    else:
        broadcast_strategy = 'none'

    set_default_logger_level('INFO')

    if args.attn_type == 'custom':
        logger.info('Using custom attention for gradient experiment')   
    nnscaler_llama_init(
        attn_type=args.attn_type,
        attn_save_path=args.attn_save_path,
    )

    ## Setup Dataset ##
    dataset = load_from_disk(args.dataset_path)
    tokenizer = get_tokenizer(args.model_id)
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

    ## Config Trainer ##
    if args.run_mode == 'compile':
        if args.runtime_ngpus is None:
            raise ValueError('runtime_ngpus must be specified in compile mode')
        runtime_ngpus = args.runtime_ngpus
    elif args.run_mode == 'run':
        world_size = int(os.getenv('WORLD_SIZE'))
        if args.runtime_ngpus is None:
            runtime_ngpus = world_size
        else:
            if args.runtime_ngpus != world_size:
                raise ValueError('runtime_ngpus must match the number of GPUs in run mode')
            runtime_ngpus = args.runtime_ngpus
            
    if runtime_ngpus % args.plan_ngpus != 0:
        raise ValueError('runtime_ngpus must be a multiple of plan_ngpus')

    compute_config = ComputeConfig(
        plan_ngpus=args.plan_ngpus,
        runtime_ngpus=runtime_ngpus,
        constant_folding=True,
        use_zero=True,
        use_end2end=True,
        # autodist config:
        # - memory constraint is set to 64GB
        # - recompute by the transformer layer in Llama
        pas_config={
            'mem_constraint': 64,
            'recompute_modules': 'LlamaDecoderLayer',
        },
    )

    model_config = ModelConfig(
        type=WrapperModel,
        args={
            'model_id': args.model_id,
            'attn_type': args.attn_type,
        },
    )

    # optimizer hyperparameters are from YaRN
    optimizer_config = OptimizerConfig(
        type=MixedPrecisionAdamW,
        args={'lr': 2e-5, 'betas': (0.9, 0.95), 'weight_decay': 0.0, 'fused': True},
        clip_gnorm=1.0,
        loss_reduction='sum',
        grad_reduction='per-token-mean',
        aggregate_outputs_fn=aggregate_outputs_fn,
    )

    dataset_config = DatasetConfig(
        type=(lambda split: dataset),
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

    # checkpoint_config = CheckpointConfig(
    #     # every_n_train_steps=1000,
    #     every_n_train_steps=5000,
    #     save_type='deduped',
    #     resume_from=(args.resume_path or 'last'),
    # )
    checkpoint_config = CheckpointConfig(
        every_n_epochs=1,
        save_type='deduped',
        resume_from=(args.resume_path or 'last'),
    )

    log_config = LogConfig(
        type=TensorBoardLogger,
        args={
            'name': args.name,
            'root_dir': './runs',
        },
    )

    trainer_args = TrainerArgs(
        pas_policy='autodist',
        precision='bf16',
        grad_accumulation_steps=4,
        seed=0,

        instance_name=args.name,
        run_mode=args.run_mode,
        max_epochs=args.n_epochs,
        max_train_steps=args.n_iter,
        enable_progress_bar=not args.disable_progressbar,
        
        compute_config=compute_config,
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        dataloader=dataloader_config,
        checkpoint=checkpoint_config,
        log=[log_config],
        
        broadcast_strategy=broadcast_strategy,
        dataset_sampler=sampler_config,   
    )

    trainer = Trainer(
        train_args=trainer_args,
        save_data_steps=args.save_step,
    )
    trainer.run()


if __name__ == '__main__':
    ## Parse Args ##
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, default='llama3-8b', help='name of the experiment')
    parser.add_argument('--run_mode', type=str, default='run', choices=['run', 'compile'], help='run or compile')
    parser.add_argument('--plan_ngpus', type=int, required=True, help='specify the scale unit size')
    parser.add_argument('--runtime_ngpus', type=int, required=True, help='specify the number of GPUs to use')
    parser.add_argument('--resume_path', type=str, default=None, help='path to dir of ckpts or the ckpt file to resume from')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to the dataset')
    parser.add_argument('--model_id', type=str, default='microsoft/llama-gpt3-8b', help='transformers model id')
    parser.add_argument('-p', '--disable_progressbar',action='store_true',help='transformers model id',)

    parser.add_argument('--n_iter', type=int, default=None, help='Number of iterations')
    parser.add_argument('--n_epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('-a', '--attn_type', type=str, default='custom', help='attention type')
    parser.add_argument('-s', '--save_step', type=int, default=1, help='Save attention data every n steps')
    parser.add_argument('--attn_save_path', type=str, default=None, help='path to save attention data')

    args = parser.parse_args()
    main(args)
