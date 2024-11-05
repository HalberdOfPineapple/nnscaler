import time
import torch
import psutil
import logging
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import torch.distributed
from torch.utils.data import DataLoader

import nnscaler
import nnscaler.utils
from nnscaler.cli.trainer import Trainer, _StepStat, TrainerArgs
from nnscaler.utils import enforce_zero_num_worker, is_running_distributed

logger = logging.getLogger(__name__)

ITERATOR_COUNTER = defaultdict(int)
def get_iter_cnt(rank: int):
    global ITERATOR_COUNTER
    return ITERATOR_COUNTER.get(rank, 0)

SAVE_ITERVAL = -1
def need_save_data(rank: int):
    global SAVE_ITERVAL
    if SAVE_ITERVAL <= 0: return False
    return get_iter_cnt(rank) % SAVE_ITERVAL == 0
    

class CustomTrainer(Trainer):
    def __init__(self,
                 argv: Optional[List[str]] = None,
                 *,
                 train_args: Optional[Union[Dict[str, Any], TrainerArgs]] = None,
                 save_data_steps: int = 1):
        """
        Custom trainer with an additional parameter.

        Args:
            argv (Optional[List[str]]): Command line arguments. If not specified, sys.argv[1:] will be used.
            train_args: A dict used to construct TrainerArgs or a TrainerArgs object itself.
            additional_param (Optional[Any]): Additional parameter for custom functionality.
        """
        # Call the parent class's initializer with the existing parameters
        super().__init__(argv=argv, train_args=train_args)
        torch.distributed.init_process_group(
            backend='nccl',
            timeout=timedelta(hours=5),
        )
        
        global SAVE_ITERVAL
        SAVE_ITERVAL = save_data_steps
        self.save_data_steps = save_data_steps

    def _train_epoch(self, epoch):
        VAL_STATUS_NO = 0     # not validated or saved
        VAL_STATUS_VAL = 1    # validated but not saved
        VAL_STATUS_SAVE = 2   # validated and saved
        has_validated = VAL_STATUS_NO   # 3 states

        resume_from_idx = self.train_status.finished_train_steps % self.total_train_steps_per_epoch
        data_iter = enumerate(self._global_batch_iterator(num_skip_first=resume_from_idx))

        max_epoch = self.max_train_steps // self.total_train_steps_per_epoch
        if self.max_train_steps % self.total_train_steps_per_epoch != 0:
            max_epoch += 1
        ndigits = len(str(max_epoch))
        epoch_format = f"0{ndigits}d"
        epoch_desc = f'Epoch {format(epoch, epoch_format)}'

        if self.rank == 0:
            progress = tqdm(
                None,
                total=self.total_train_steps_per_epoch,
                initial=resume_from_idx,
                desc=epoch_desc,
                disable=not self.train_args.enable_progress_bar,
            )
        else:
            progress = None

        step_stat: Optional[_StepStat] = None
        for i, batches in data_iter:
            idx = i + resume_from_idx
            
            global ITERATOR_COUNTER
            ITERATOR_COUNTER[self.rank] = idx
            # print(f"|{__name__}| rank={self.rank}, ITERATOR_COUNTER[self.rank]={ITERATOR_COUNTER[self.rank]}")

            if self.rank == 0:
                # looks manually update progress bar is easier
                # than using tqdm directly
                # the difference is we update progress bar at the beginning of the loop
                # instead of the end of the loop
                progress.update(1)
            step_start_at = time.perf_counter()
            step_stat = _StepStat()
            step_metrics = {}
            has_validated = VAL_STATUS_NO
            num_batches = len(batches)
            batches, is_dummy_batch = self._fix_batches(batches)

            self.model.train()

            self.hook.before_zero_grad(self)
            self.optimizer.zero_grad()
            self.hook.after_zero_grad(self)

            self.hook.on_train_step_start(self, batches[:num_batches], idx)
            losses = self.model.train_step(batches, is_dummy_batch)
            self.hook.on_train_step_end(self, losses[:num_batches], batches[:num_batches], idx)

            aggregate_outputs = self.train_args.resolved_aggregate_outputs_fn or self.aggregate_outputs
            aggregated_outputs = aggregate_outputs(losses[:num_batches], self.sync_group)
            if self.train_args.optimizer.loss_reduction == 'mean':
                loss = aggregated_outputs.loss_sum / aggregated_outputs.num_batches
            else:
                loss = aggregated_outputs.loss_sum
            step_stat.train_loss = loss
            self.hook.after_aggregate_train_step_outputs(self, aggregated_outputs, loss, idx)

            self.hook.before_sync_grad(self)
            # actually `sync_shard_grad` is no-op here
            # because trainer only supports end2end model
            # and syncing grad in end2end model is done in `_train_step`.
            self.optimizer.sync_shard_grad()
            self.hook.after_sync_grad(self)

            # scale gradients
            multiplier = self.train_args.scaling_factor
            if self.train_args.optimizer.grad_reduction == 'sum':
                # do nothing. `multiplier` is already correct
                pass
            elif self.train_args.optimizer.grad_reduction == 'mean':
                if not aggregated_outputs.num_batches:
                    raise RuntimeError("`aggregate_outputs` doesn't set `num_batches` field")
                multiplier /= aggregated_outputs.num_batches
            else:
                assert self.train_args.optimizer.grad_reduction == 'per-token-mean'
                if not aggregated_outputs.num_tokens:
                    raise RuntimeError("`aggregate_outputs` doesn't set `num_tokens` field")
                multiplier /= aggregated_outputs.num_tokens
            self.optimizer.scale_grads(multiplier)

            # clip gradients
            self.hook.before_gnorm_clip(self)
            if self.train_args.optimizer.clip_gnorm:
                step_stat.gnorm = self.optimizer.clip_gnorm(self.train_args.optimizer.clip_gnorm)
            else:
                step_stat.gnorm = self.optimizer.clip_gnorm()
            self.hook.after_gnorm_clip(self, step_stat.gnorm)
            step_stat.gnorm = step_stat.gnorm.item()

            # update parameters
            step_stat.lr = self.optimizer.param_groups[0]['lr']
            self.hook.before_optimizer_step(self)
            self.optimizer.step()
            self.hook.after_optimizer_step(self)
            if self.lr_scheduler and self.train_args.lr_scheduler.interval == 'step':
                self.lr_scheduler.step()

            self.train_status.finished_train_steps += 1
            self._log_mem_stats(tag='train')
            step_metrics = {k:v for k, v in asdict(step_stat).items() if v is not None}
            step_metrics['train_wall'] = time.perf_counter() - step_start_at
            self.log_metrics(step_metrics, tag='train')
            if self.rank == 0:
                progress.set_postfix(step_metrics)
                if self.train_args.enable_log_progress \
                    and self.train_status.finished_train_steps % self.train_args.log_progress_every_n_train_steps == 0:
                    logger.info(self._format_metrics(epoch_desc, idx + 1, step_metrics))
                    step_metrics = {}

            # validate and save checkpoint
            if self.train_args.checkpoint.every_n_train_steps and \
                self.train_status.finished_train_steps % self.train_args.checkpoint.every_n_train_steps == 0:
                self._validate_and_save(step_stat)
                has_validated = VAL_STATUS_SAVE

            # max_train_steps is reached
            if self.train_status.finished_train_steps >= self.max_train_steps:
                if step_metrics and self.train_args.enable_log_progress:
                    logger.info(self._format_metrics(epoch_desc, idx + 1, step_metrics))
                    step_metrics = {}
                if not has_validated:
                    self._validate_and_save(step_stat)
                    has_validated = VAL_STATUS_SAVE
                if self.rank == 0:
                    # disable refresh the progress bar to avoid redundant progress bar
                    progress.leave = False
                    progress.close()
                break

            if not has_validated and self.train_args.val_every_n_train_steps and \
                self.train_status.finished_train_steps % self.train_args.val_every_n_train_steps == 0:
                self._validate(step_stat)
                has_validated = VAL_STATUS_VAL

            # time.sleep(1)
        else:
            # Do per-epoch operations here.
            # if the loop exits with `break` (max_train_steps is reached)
            # those operations have done in the loop
            if step_stat is None:
                return  # no train step runs. Nothing to do.
            if has_validated < VAL_STATUS_SAVE \
                and self.train_args.checkpoint.every_n_epochs \
                and (epoch + 1) % self.train_args.checkpoint.every_n_epochs == 0:
                self._validate_and_save(step_stat)
                has_validated = VAL_STATUS_SAVE
            if not has_validated and self.train_args.val_every_n_epochs \
                and (epoch + 1) % self.train_args.val_every_n_epochs == 0:
                self._validate(step_stat)
                has_validated = VAL_STATUS_VAL