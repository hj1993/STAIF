# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.utils.versions import require_version
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.utils import RunningMoments, selective_log_softmax

from swift.llm import to_device
from swift.utils import get_logger
from ..mixin import DataLoaderMixin, SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFDPOTrainer.__init__
logger = get_logger()


def new_gather_function(tensor):
    tensor_list = gather_object([tensor])
    tensor_list = [t[None] if t.ndim == 0 else t for t in tensor_list]
    return torch.concat(to_device(tensor_list, tensor.device), dim=0)


class DPOTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HFDPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import FDivergenceConstants
        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        if 'loss_weights' in DPOConfig.__dict__:
            # trl >= 0.20
            self.loss_type = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
            self.loss_weights = args.loss_weights
        else:
            self.loss_type = args.loss_type

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        for loss_type in loss_types:
            if (loss_type in ['hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'apo_zero', 'apo_down']
                    and args.label_smoothing > 0):
                warnings.warn(
                    f'You are using the {loss_type} loss type that does not support label smoothing. The '
                    '`label_smoothing` parameter will be ignored. '
                    'Set `label_smoothing` to `0.0` to remove this warning.',
                    UserWarning,
                )
            if loss_type == 'kto_pair':
                raise ValueError('Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.')

        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = args.ref_adapter_name
        self.model_adapter_name = None
        self.reference_free = args.reference_free
        self.use_weighting = False

        super().__init__(model, ref_model, *_args, **kwargs)

        if 'bco_pair' in loss_types:
            self.running = RunningMoments(self.accelerator)
        if self.args.ld_alpha is not None:
            require_version('trl>=0.18', '`ld_alpha` requires that "trl>=0.18".')
        if self.template.packing:
            self.accelerator.gather_for_metrics = new_gather_function

    @contextmanager
    def null_ref_context(self):
        with self.accelerator.unwrap_model(self.model).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or 'default')

    @staticmethod
    def _split_rejected_by_group(rejected_tensor: torch.Tensor, num_rejected: torch.Tensor) -> List[torch.Tensor]:
        groups = []
        start = 0
        for size in num_rejected.tolist():
            end = start + size
            groups.append(rejected_tensor[start:end])
            start = end
        return groups

    def multi_negative_dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_logratios = chosen_logps - (0 if self.reference_free else ref_chosen_logps)
        rejected_logratios = rejected_logps - (0 if self.reference_free else ref_rejected_logps)

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        grouped_rejected_rewards = self._split_rejected_by_group(rejected_rewards, num_rejected)

        losses = []
        grouped_mean_rejected_rewards = []
        for chosen_reward, sample_rejected_rewards in zip(chosen_rewards, grouped_rejected_rewards):
            competitor_scores = sample_rejected_rewards - chosen_reward
            losses.append(torch.logsumexp(torch.cat([competitor_scores.new_zeros((1, )), competitor_scores]), dim=0))
            grouped_mean_rejected_rewards.append(sample_rejected_rewards.mean())
        losses = torch.stack(losses)
        grouped_mean_rejected_rewards = torch.stack(grouped_mean_rejected_rewards)
        return losses, chosen_rewards, grouped_mean_rejected_rewards

    def compute_ref_log_probs(self, batch):
        with torch.no_grad(), self.null_ref_context():
            if self.ref_model is None:
                ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
        return ref_model_output['chosen_logps'], ref_model_output['rejected_logps']

    def get_batch_loss_metrics(self, model, batch, train_eval: str = 'train'):
        metrics = {}
        model_output = self.concatenated_forward(model, batch)
        if 'ref_chosen_logps' in batch and 'ref_rejected_logps' in batch:
            ref_chosen_logps = batch['ref_chosen_logps']
            ref_rejected_logps = batch['ref_rejected_logps']
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        num_rejected = batch.get('num_rejected')
        losses = 0
        chosen_rewards = 0
        rejected_rewards = 0
        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        for idx, loss_type in enumerate(loss_types):
            if num_rejected is not None and torch.any(num_rejected != 1):
                if loss_type != 'sigmoid':
                    raise ValueError(f'Multi-negative DPO currently only supports `sigmoid`, got `{loss_type}`.')
                _losses, _chosen_rewards, _rejected_rewards = self.multi_negative_dpo_loss(
                    model_output['chosen_logps'],
                    model_output['rejected_logps'],
                    ref_chosen_logps,
                    ref_rejected_logps,
                    num_rejected,
                )
            else:
                _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                    model_output['chosen_logps'],
                    model_output['rejected_logps'],
                    ref_chosen_logps,
                    ref_rejected_logps,
                )
            weight = self.loss_weights[idx] if getattr(self, 'loss_weights', None) else 1.0
            losses = losses + _losses * weight
            chosen_rewards = chosen_rewards + _chosen_rewards * weight
            rejected_rewards = rejected_rewards + _rejected_rewards * weight

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output['nll_loss']
        if self.use_weighting:
            losses = losses * model_output['policy_weights']
        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output['aux_loss']

        prefix = 'eval_' if train_eval == 'eval' else ''
        metrics[f'{prefix}rewards/chosen'] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f'{prefix}rewards/rejected'] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f'{prefix}rewards/accuracies'] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f'{prefix}rewards/margins'] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item())
        metrics[f'{prefix}logps/chosen'] = (
            self.accelerator.gather_for_metrics(model_output['chosen_logps']).detach().mean().item())
        metrics[f'{prefix}logps/rejected'] = (
            self.accelerator.gather_for_metrics(model_output['rejected_logps']).detach().mean().item())
        metrics[f'{prefix}logits/chosen'] = (
            self.accelerator.gather_for_metrics(model_output['mean_chosen_logits']).detach().mean().item())
        metrics[f'{prefix}logits/rejected'] = (
            self.accelerator.gather_for_metrics(model_output['mean_rejected_logits']).detach().mean().item())
        if self.args.rpo_alpha is not None:
            metrics[f'{prefix}nll_loss'] = (
                self.accelerator.gather_for_metrics(model_output['nll_loss']).detach().mean().item())
        if self.aux_loss_enabled:
            metrics[f'{prefix}aux_loss'] = (
                self.accelerator.gather_for_metrics(model_output['aux_loss']).detach().mean().item())

        return losses.mean(), metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self.template.forward_context(self.model, inputs):
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval='train')
        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval='train')
        if return_outputs:
            return loss, metrics
        return loss

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_ref_model: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch = batch.copy()

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(batch)
        if self.aux_loss_enabled:
            batch['output_router_logits'] = True
        labels = batch.pop('labels', None)
        if self.is_encoder_decoder:
            batch['labels'] = labels
        text_position_ids = batch.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = batch.get('position_ids')
        outputs = model(**batch, use_cache=False)
        all_logits = outputs.logits

        if all_logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            all_logits = all_logits[:, -labels.shape[1]:]

        if not self.is_encoder_decoder and self.template.sequence_parallel_size == 1:
            # Shift so that tokens < n predict n
            labels = torch.roll(labels, shifts=-1, dims=1)
        per_token_logps, mean_all_logits, loss_mask = self.get_per_token_logps(
            all_logits, labels, label_pad_token_id=self.label_pad_token_id)
        origin_per_token_logps = per_token_logps

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        if 'ipo' in loss_types:
            size_completion = loss_mask.sum(dim=-1)
            per_token_logps = per_token_logps / size_completion

        output = {}
        if self.template.padding_free:
            cu_seqlens = self.get_cu_seqlens(text_position_ids, batch.get('logits_to_keep'))
            num_chosen = int(batch.get('num_chosen', 0) or batch.get('num_rejected', labels.new_zeros((labels.shape[0] // 2, )).shape[0]) or 0)
            if not num_chosen:
                num_chosen = labels.shape[0] // 2
            total_examples = cu_seqlens.shape[0] - 1
            all_logps = per_token_logps.new_zeros((total_examples, ))
            completion_lengths = (cu_seqlens[1:] - cu_seqlens[:-1])
            chosen_lengths = completion_lengths[:num_chosen]
            num_rejected = batch.get('num_rejected')
            if num_rejected is not None and torch.any(num_rejected != 1):
                rejected_lengths = completion_lengths[num_chosen:]
                public_lengths = []
                start = 0
                for i, size in enumerate(num_rejected.tolist()):
                    end = start + size
                    public_lengths.extend([torch.min(chosen_lengths[i], rejected_lengths[start:end].min())] * size)
                    start = end
                public_lengths = torch.stack(public_lengths)
            else:
                rejected_lengths = completion_lengths[num_chosen:]
                public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper

            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                length = end - start
                if i < num_chosen:
                    public_length = public_lengths[i] if public_lengths.ndim > 0 else public_lengths
                else:
                    rej_idx = i - num_chosen
                    public_length = public_lengths[rej_idx if public_lengths.shape[0] != num_chosen else rej_idx % num_chosen]
                if self.args.ld_alpha is not None and not is_ref_model and length > public_length:
                    front_logps = per_token_logps[:, start:start + public_length].sum()
                    rear_logps = per_token_logps[:, start + public_length:end].sum()
                    all_logps[i] = front_logps + self.args.ld_alpha * rear_logps
                else:
                    all_logps[i] = per_token_logps[:, start:end].sum()
            num_tokens = cu_seqlens[num_chosen]
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['chosen_logps'] = all_logps[:num_chosen]
            output['rejected_logps'] = all_logps[num_chosen:]
            output['mean_chosen_logits'] = mean_all_logits[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['mean_rejected_logits'] = mean_all_logits[:, num_tokens:][loss_mask[:, num_tokens:]].mean()
        else:
            num_chosen = int(batch.get('num_chosen', 0) or batch.get('num_rejected', labels.new_zeros((labels.shape[0] // 2, )).shape[0]) or 0)
            if not num_chosen:
                num_chosen = labels.shape[0] // 2
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:num_chosen][loss_mask[:num_chosen]].mean()
            if self.args.ld_alpha is not None and not is_ref_model:
                completion_lengths = loss_mask.sum(dim=1)

                chosen_lengths = completion_lengths[:num_chosen]
                rejected_lengths = completion_lengths[num_chosen:]
                num_rejected = batch.get('num_rejected')
                if num_rejected is not None and torch.any(num_rejected != 1):
                    chosen_public_lengths = []
                    rejected_public_lengths = []
                    start = 0
                    for i, size in enumerate(num_rejected.tolist()):
                        end = start + size
                        public_length = torch.min(chosen_lengths[i], rejected_lengths[start:end].min())
                        chosen_public_lengths.append(public_length)
                        rejected_public_lengths.extend([public_length] * size)
                        start = end
                    public_lengths = torch.cat(
                        [torch.stack(chosen_public_lengths), torch.stack(rejected_public_lengths)], dim=0)
                else:
                    public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
                    public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

                seq_len = per_token_logps.size(1)
                text_position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

                ld_mask = text_position_ids < public_lengths.unsqueeze(1)
                mask = text_position_ids < completion_lengths.unsqueeze(1)

                front_mask = (ld_mask & mask).float()
                rear_mask = (~ld_mask & mask).float()
                front_logps = (per_token_logps * front_mask).sum(dim=1)
                rear_logps = (per_token_logps * rear_mask).sum(dim=1)

                all_logps = front_logps + self.args.ld_alpha * rear_logps
            else:
                all_logps = per_token_logps.sum(-1)
            output['chosen_logps'] = all_logps[:num_chosen]
            output['rejected_logps'] = all_logps[num_chosen:]
            output['mean_chosen_logits'] = mean_all_logits[:num_chosen][loss_mask[:num_chosen]].mean()
            output['mean_rejected_logits'] = mean_all_logits[num_chosen:][loss_mask[num_chosen:]].mean()
        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss
        return output

    @staticmethod
    def get_per_token_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id=-100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(f'Logits (batch and sequence length dim) {logits.shape[:-1]}'
                             'and labels must have the same shape {labels.shape}')
        loss_mask = labels != label_pad_token_id
        labels = labels.clone()
        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        return per_token_logps, logits.mean(-1), loss_mask

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)

    def prediction_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().prediction_step(model, inputs, *args, **kwargs)
