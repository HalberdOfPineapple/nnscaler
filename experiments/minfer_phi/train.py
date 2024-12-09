#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import os
import yaml
import torch
import datasets
import argparse
import huggingface_hub
from typing import Dict, List
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

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

from minfer_modules import ExprMInferConfig as MInferenceConfig, ExprMInference as MInference
from chunk_linear_cross_entropy import chunk_linear_cross_entropy
from custom_trainer import CustomTrainer as Trainer # from nnscaler.cli.trainer import Trainer

import logging
logger = logging.getLogger(__name__)

IGNORE_IDX = -100
MINFER_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


# define a enumerate class for MInfer type
class MInferType:
    BASELINE: str = "baseline"
    MF_DB: str = "mf_db"
    DF_MB: str = "df_mb"
    MF_MB: str = "mf_mb"


def nnscaler_phi_init(attn_type: str='flash', attn_save_path: str=None):
    from phi3 import PHI3_ATTENTION_CLASSES
    from modeling_modifier import NNScalerPhiFlashAttention2
    
    PHI3_ATTENTION_CLASSES["flash_attention_2"] = NNScalerPhiFlashAttention2

def minfer_phi_init():
    from phi3 import PHI3_ATTENTION_CLASSES
    from minfer_modifier_v2 import MInferAttention

    PHI3_ATTENTION_CLASSES["flash_attention_2"] = MInferAttention

def get_tokenizer(tokenizer_name_or_path,
                  model_max_length=None,
                  default_bos_token="<s>",
                  default_eos_token="</s>",
                  default_pad_token="[PAD]",
                  default_unk_token="<unk>"):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
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

def get_module_path(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    module_path = str(model.__class__.__module__)
    del model

    return module_path


def minfer_patch_setup(model, minfer_config: MInferenceConfig):
    if minfer_config.attn_type != "vllm":
            model.config.starting_layer = minfer_config.starting_layer
            model.config.config_path = minfer_config.config_path

    if minfer_config.attn_type == "minference":
        model.config.is_search = minfer_config.is_search

    elif minfer_config.attn_type == "minference_with_dense":
        model.config.dense = True

    elif minfer_config.attn_type == "dilated1":
        model.config.dilated1 = True

    elif minfer_config.attn_type == "static":
        model.config.static_pattern = True
    elif minfer_config.attn_type == "dilated2":
        model.config.dilated2 = True

    elif minfer_config.attn_type == "streaming":
        model.config.streaming = True
        model.config.streaming_kwargs = {
            "n_local": 3968,
            "n_init": 128,
            **minfer_config.attn_kwargs,
        }
    elif minfer_config.attn_type == "hf":
        pass
    else:
        raise ValueError(
            f"The attention type {minfer_config.attn_type} is currently not supported for training"
        )
    return model

class BaselineModel(torch.nn.Module):
    def __init__(self, model_id, config_path: str=None, selected_layers: List[int]=[]):
        super().__init__()
        from phi3 import Phi3ForCausalLM

        if not config_path:
            self.model = Phi3ForCausalLM.from_pretrained(
                model_id,
                attn_implementation='flash_attention_2'
            )
        else:
            model_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
            model_config._attn_implementation = 'flash_attention_2'
            self.model = Phi3ForCausalLM.from_pretrained(
                model_id,
                config=model_config,
            )
        if len(selected_layers) > 0:
            self.model.model.layers = torch.nn.ModuleList([self.model.model.layers[i] for i in selected_layers])
            
        print(f'{__name__} BaselineModel Selt-Attention Class: {self.model.model.layers[0].self_attn.__class__.__name__}')

    def forward(self, samples):
        with torch.autocast(device_type="cuda", dtype=self.model.config.torch_dtype):
            outputs = self.model.model(
                input_ids=samples['net_input']['src_tokens'],
                use_cache=False,
                return_dict=False,
            )
            hidden_states = outputs[0]
            losses = chunk_linear_cross_entropy(hidden_states, self.model.lm_head.weight, samples['target'], IGNORE_IDX, 1024)
            loss = torch.sum(losses)

        return loss, loss.data, samples['ntokens'], samples['nsentences']

class MInferModel(BaselineModel):
    def __init__(self, model_id, config_path: str=None, minfer_config: Dict={}, selected_layers: List[int]=[]):
        super().__init__(
            model_id=model_id,
            config_path=config_path,
            selected_layers=selected_layers,
        )

        minfer_attn_type = minfer_config.pop('attn_type', 'minference')
        minfer = MInference(
            attn_type=minfer_attn_type,
            model_name=model_id,
            **minfer_config,
        )
        minfer_config: MInferenceConfig = minfer.config
        
        self.model = minfer_patch_setup(self.model, minfer_config)

        Attention = self.model.model.layers[0].self_attn.__class__
        def update_module(m):
            if isinstance(m, Attention):
                m.init_minference_parameters()
        self.model.apply(update_module)

        # self.model = minfer_phi_init(self.model, model_id, minfer_config)


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

    # if os.environ.get("DEBUG_MODE") == "1":
    #     import debugpy
    #     base_port = 5678
    #     port = base_port + int(os.environ["LOCAL_RANK"])
    #     debugpy.listen(("0.0.0.0", port))
    #     print(f"Waiting for debugger attach on rank {os.environ['LOCAL_RANK']} (port {port})...")
    #     debugpy.wait_for_client()

    if args.minfer_type == MInferType.BASELINE:
        print(f"{__name__} | (Expr 0) Using Baseline Model...")
        nnscaler_phi_init()
    else:
        print(f"{__name__} | (Expr 3) Using MInference-equipped Model ...")
        minfer_phi_init()

    minfer_config_path = os.path.join(MINFER_CONFIG_DIR, f'{args.name}.yaml')
    if not os.path.exists(minfer_config_path):
        minfer_config = {}
    else:
        with open(minfer_config_path, 'r') as f:
            minfer_config = yaml.safe_load(f)

    if args.run_mode == 'run':
        broadcast_strategy = 'all'
    else:
        broadcast_strategy = 'none'

    set_default_logger_level('INFO')
    

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
        pas_config={
            # 'mem_constraint': 64, # - memory constraint is set to 64GB
            'recompute_modules': 'Phi3DecoderLayer',
        },
    )

    model_args = {
        'model_id': args.model_id,
        'config_path': args.model_config_path,
        'selected_layers': args.selected_layers,
    }
    if args.minfer_type != 'baseline': model_args['minfer_config'] = minfer_config
    model_config = ModelConfig(
        type=MInferModel if args.minfer_type != 'baseline' else BaselineModel,
        args=model_args,
    )

    # optimizer hyperparameters are from YaRN
    optimizer_config = OptimizerConfig(
        type=MixedPrecisionAdamW,
        args={
            'lr': 2e-5, 
            # 'lr': 1e-6,
            'betas': (0.9, 0.95), 
            'weight_decay': 0.0, 
            'fused': True
        },
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

    checkpoint_config = CheckpointConfig(
        save_dir=args.ckpt_save_dir if args.ckpt_save_dir else f'./checkpoints_{args.name}',
        every_n_epochs=args.ckpt_n_epoch,
        every_n_train_steps=args.ckpt_n_step,
        save_type='deduped',
        resume_from=(args.ckpt_save_dir or 'last') if args.check_resume else None,
    )

    log_config = LogConfig(
        type=TensorBoardLogger,
        args={
            'name': args.name,
            'root_dir': args.tf_log_dir or f'./runs_{args.name}',
        },
    )

    scaling_factor: int = runtime_ngpus // args.plan_ngpus
    trainer_args = TrainerArgs(
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        grad_accumulation_steps=args.global_batch_size // (args.micro_batch_size * scaling_factor),

        pas_policy='autodist',
        precision='bf16',
        seed=0,
        gen_reuse=args.reuse_type,

        gen_savedir=args.compile_save_path,
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
        save_data_steps=args.attn_save_step,
    )
    trainer.run()

def print_args(args: argparse.Namespace):
    print("=" * 80)
    print(f"Start Experiment:\t{args.name}")
    print(f"Reuse Type:\t{args.reuse_type}")
    print(f"Run Mode:\t{args.run_mode}")
    print(f"Total number of GPUs:\t{args.runtime_ngpus}")
    print(f"GPU unit size:\t{args.plan_ngpus}")
    print(f"Model ID:\t{args.model_id}")

    print('-' * 40)
    if args.n_iter:
        print(f"Number of Iterations:\t{args.n_iter}")
    else:
        print(f"Number of Epochs:\t{args.n_epochs}")

    print(f'Global Batch Size:\t{args.global_batch_size}')
    print(f'Micro Batch Size:\t{args.micro_batch_size}')

    scaling_factor = args.runtime_ngpus // args.plan_ngpus
    grad_accu_step = args.global_batch_size // (args.micro_batch_size * scaling_factor)
    print(f"Scaling Factor (INFERRED):\t{scaling_factor}")
    print(f"Gradient Accumulation Steps (INFERRED):\t{grad_accu_step}")
    print(f"Save Attention Data Every {args.attn_save_step} Steps")

    print('-' * 40)
    print(f"Model Config Path:\t{args.model_config_path}")
    print(f'MInferenece Config Path:\t{args.minfer_config_path}')
    print(f"Compile Save Path:\t{args.compile_save_path}")
    print(f"Attention Save Path:\t{args.attn_save_path}")
    print(f"Tensorboard Log Path:\t{args.tf_log_dir}")
    print(f"Checkpoint Save Path:\t{args.ckpt_save_dir}")
    print(f"Resume from Checkpoint:\t{args.check_resume}")
    if args.ckpt_n_step:
        print(f"Checkpoint Save Every {args.ckpt_n_step} Steps")
    else:
        print(f"Checkpoint Save Every {args.ckpt_n_epoch} Epochs")
    print("=" * 80)

if __name__ == '__main__':
    ## Parse Args ##
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, default='phi-grad', help='name of the experiment')
    parser.add_argument('--minfer_type', type=str, default=MInferType.BASELINE, choices=MInferType.__dict__.values(), help='minference type')
    parser.add_argument('--reuse_type', type=str, default='match', choices=['match', 'override', 'moo', 'graph'], help='reuse type')
    parser.add_argument('--run_mode', type=str, default='run', choices=['run', 'compile'], help='run or compile')
    parser.add_argument('--plan_ngpus', type=int, required=True, help='specify the scale unit size')
    parser.add_argument('--runtime_ngpus', type=int, required=True, help='specify the number of GPUs to use')
    
    parser.add_argument('--n_iter', type=int, default=0, help='Number of iterations')
    parser.add_argument('--n_epochs', type=int, default=0, help='Number of epochs')
    parser.add_argument('--global_batch_size', type=int, default=4, help='global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=1, help='micro batch size')

    parser.add_argument('--model_id', type=str, default='microsoft/Phi-3-mini-4k-instruct', help='transformers model id')
    parser.add_argument('--model_config_path', type=str, default=None, help='path to the model config')
    parser.add_argument('-s', '--attn_save_step', type=int, default=1, help='Save attention data every n steps')

    parser.add_argument('--minfer_config_path', type=str, default=None, help='path to minference config')
    parser.add_argument('--compile_save_path', type=str, default='./.nnscaler', help='path to save compiled code')
    parser.add_argument('--attn_save_path', type=str, default=None, help='path to save attention data')
    parser.add_argument('--tf_log_dir', type=str, default=None, help='path to save tensorboard logs')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to the dataset')
    parser.add_argument('--check_resume', action='store_true', help='whether to resume from checkpoint')

    parser.add_argument('--ckpt_save_dir', type=str, default=None, help='path to save checkpoints')
    parser.add_argument('--ckpt_n_epoch', type=int, default=1, help='save checkpoint every n epochs')
    parser.add_argument('--ckpt_n_step', type=int, default=0, help='save checkpoint every n steps')

    parser.add_argument('-p', '--disable_progressbar',action='store_true',help='transformers model id',)

    # add selected layers as the argument
    parser.add_argument('--selected_layers', type=str, default='[]', help='selected layers')

    args = parser.parse_args()
    print_args(args)

    if args.ckpt_n_epoch <= 0: args.ckpt_n_epoch = None
    if args.ckpt_n_step <= 0: args.ckpt_n_step = None

    if args.n_iter <= 0: args.n_iter = None
    if args.n_epochs <= 0: args.n_epochs = None

    args.selected_layers = eval(args.selected_layers)
    main(args)

