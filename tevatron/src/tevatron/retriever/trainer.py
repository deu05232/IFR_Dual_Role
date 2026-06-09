import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from modeling import EncoderModel
from transformers import AutoTokenizer


import logging
logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):
    def __init__(self, dont_shuffle, train_group_size=None, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"
        self.train_group_size = train_group_size

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage  = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(TevatronTrainer, self)._get_train_sampler()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # In-group nDCG: rank only the train_group_size passages of each query.
        # K=1 (plain InfoNCE). For JointLH variants this is overridden.
        query, passage = inputs[0], inputs[1]
        G = self.train_group_size
        assert G is not None, "train_group_size must be set on the trainer for eval."

        with torch.no_grad():
            q_reps = model.encode_query(query)
            p_reps = model.encode_passage(passage)

        B = q_reps.size(0)
        p_reps = p_reps.view(B, G, -1)
        scores = (q_reps.unsqueeze(1) * p_reps).sum(-1)               # [B, G]

        labels = torch.zeros(B, G, device=scores.device, dtype=scores.dtype)
        labels[:, 0] = 1.0
        return (None, scores, labels)



class JointLHTevatronTrainer(TevatronTrainer):
    def __init__(self, num_positives, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_positives = num_positives

    def compute_loss(self, model, inputs):
        query, passage, instruct_flag = inputs
        return model(
            query=query,
            passage=passage,
            instruct_flag=instruct_flag,
            num_positives=self.num_positives,
        ).loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # In-group nDCG with has_instruction split.
        # inputs: (query, passage, instruct_flag) for JointLH,
        #         (query, passage, instruct_flag, margin_flag, only_query) for JointLHMargin.
        query, passage, instruct_flag = inputs[0], inputs[1], inputs[2]
        G = self.train_group_size
        # Eval K follows the dev dataset's num_positives (which may be pinned
        # independently of training), so pos_mask matches the dev group layout.
        K = getattr(self.eval_dataset, "num_positives", self.num_positives)
        assert G is not None, "train_group_size must be set on the trainer for eval."

        with torch.no_grad():
            q_reps = model.encode_query(query)
            p_reps = model.encode_passage(passage)

        B = q_reps.size(0)
        p_reps = p_reps.view(B, G, -1)
        scores = (q_reps.unsqueeze(1) * p_reps).sum(-1)               # [B, G]

        flag = instruct_flag.to(scores.device).long()
        k_per = torch.where(flag == 0, K, 1)                          # [B]
        arange_G = torch.arange(G, device=scores.device).unsqueeze(0)
        pos_mask = (arange_G < k_per.unsqueeze(1)).to(scores.dtype)   # [B, G]
        # Pack flag as last column so compute_metrics can split by has_instruction.
        labels = torch.cat([pos_mask, flag.to(scores.dtype).unsqueeze(1)], dim=1)  # [B, G+1]
        return (None, scores, labels)


class JointLHMarginTevatronTrainer(JointLHTevatronTrainer):
    """Trainer for JointLH + instruction-margin auxiliary loss.

    Inputs from TrainJointLHMarginCollator are
        (query, passage, instruct_flag, margin_flag, only_query)
    where only_query is always a fixed-shape [B, ...] tokenized batch
    (one entry per sample, matching `query`), so DDP all-gather of
    only_q_reps in the model produces a uniform tensor across ranks.
    Margin-flagged samples are selected inside the model's margin-loss
    computation via margin_flag.
    """

    def compute_loss(self, model, inputs):
        query, passage, instruct_flag, margin_flag, only_query = inputs
        return model(
            query=query,
            passage=passage,
            instruct_flag=instruct_flag,
            num_positives=self.num_positives,
            margin_flag=margin_flag,
            only_query=only_query,
        ).loss


class SubsetJointLHTevatronTrainer(TevatronTrainer):
    """Trainer for subset JointLH / LSEPair.

    Inputs are (query, passage, num_positives) where num_positives is a [B] long
    tensor describing the per-query positive count.
    """

    def compute_loss(self, model, inputs):
        query, passage, num_positives = inputs
        return model(
            query=query,
            passage=passage,
            num_positives=num_positives,
        ).loss
