import logging
import os
import sys
import torch

from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
    
from dataset import (
    JointLHTrainDataset,
    JointLHDistinctBatchTrainDataset,
    RandLHDistinctBatchTrainDataset,
    MultiposValidDataset
)

from collator import (
    TrainCollator,
    TrainJointLHCollator,
    TrainJointLHMarginCollator,
    TrainSubsetJointLHCollator,
)

from modeling import (
    DenseModel,
    DenseJointLHModel,
)

from trainer import (
    TevatronTrainer,
    JointLHTevatronTrainer,
)

import torch.distributed as dist
################################

from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer

import numpy as np

logger = logging.getLogger(__name__)


def _ndcg_at_k(labels, scores, k):
    """Mean linear-gain NDCG@k over rows. Equivalent to
    sklearn.metrics.ndcg_score(labels, scores, k=k) for continuous scores
    (no sklearn dependency).

    labels, scores: [N, G] arrays. Rows whose ideal DCG is 0 (no relevant
    item) contribute 0, matching sklearn.
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n, g = scores.shape
    k = min(k, g)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    order = np.argsort(-scores, axis=1)[:, :k]
    gains = np.take_along_axis(labels, order, axis=1)
    dcg = (gains * discounts).sum(axis=1)
    ideal = -np.sort(-labels, axis=1)[:, :k]
    idcg = (ideal * discounts).sum(axis=1)
    ndcg = np.zeros(n)
    nz = idcg > 0
    ndcg[nz] = dcg[nz] / idcg[nz]
    return float(ndcg.mean())


# Dev evaluation always uses a fixed number of positives per query, independent
# of the training num_positives. The JointLH eval (prediction_step) reads K back
# from the dev dataset, so the pos_mask stays consistent with this.
DEV_NUM_POSITIVES = 3


def _make_dev_dataset(ds_cls, data_args):
    """Instantiate dev dataset by pointing data_args at the dev data.

    ``dev_dataset_path`` is interpreted the same way the *train* set was
    specified, so the loader resolves it identically:
      - Hub-name mode (train uses --dataset_name with no --dataset_path):
        dev_dataset_path is a Hub dataset repo name -> override dataset_name,
        keep data_files (dataset_path) unset.
      - local/data_files mode (train uses --dataset_path): dev_dataset_path is
        a local data file/dir -> override dataset_path.

    Also pins num_positives to DEV_NUM_POSITIVES for the dev dataset only; all
    overridden fields are restored afterwards so training is unaffected.

    Returns None if data_args.dev_dataset_path is not set, so HF Trainer
    skips evaluation.
    """
    if not data_args.dev_dataset_path:
        return None
    saved_name = data_args.dataset_name
    saved_path = data_args.dataset_path
    saved_split = data_args.dataset_split
    saved_num_positives = data_args.num_positives
    if data_args.dataset_path is None:
        # train loaded from a Hub repo name -> dev path is also a repo name
        data_args.dataset_name = data_args.dev_dataset_path
    else:
        data_args.dataset_path = data_args.dev_dataset_path
    data_args.dataset_split = data_args.dev_dataset_split
    data_args.num_positives = DEV_NUM_POSITIVES
    try:
        dev = ds_cls(data_args)
    finally:
        data_args.dataset_name = saved_name
        data_args.dataset_path = saved_path
        data_args.dataset_split = saved_split
        data_args.num_positives = saved_num_positives
    return dev


def _compute_ndcg_plain(eval_pred):
    """Plain InfoNCE eval: K=1, no instruct_flag split."""
    scores, labels = eval_pred  # [N, G], [N, G]
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return {
        "ndcg@10": _ndcg_at_k(labels, scores, 10),
        "ndcg@5":  _ndcg_at_k(labels, scores, 5),
    }


def _compute_ndcg_jointlh(eval_pred):
    """JointLH eval: labels last column is instruct_flag; split by it."""
    scores, labels = eval_pred  # [N, G], [N, G+1]
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos_mask = labels[:, :-1]
    flag = labels[:, -1].astype(int)

    out = {}
    for name, sel in [("q", flag == 0), ("q_inst", flag == 1)]:
        if sel.sum() == 0:
            continue
        s, l = scores[sel], pos_mask[sel]
        out[f"ndcg@10_{name}"] = _ndcg_at_k(l, s, 10)
        out[f"ndcg@5_{name}"]  = _ndcg_at_k(l, s, 5)
    out["ndcg@10"] = _ndcg_at_k(pos_mask, scores, 10)
    return out


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    print(f"Tokenizer name is {model_args.tokenizer_name}")
    # print(tokenizer)
        
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    effective_bs = training_args.per_device_train_batch_size * world_size


    
    if "RandLH" in training_args.output_dir:
        # RandLH 및 일반적인 InfoNCE 학습
        print("***************Using Dense Model***************")
        model = DenseModel.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        train_dataset = RandLHDistinctBatchTrainDataset(data_args, effective_batch_size=effective_bs)
        dev_dataset = _make_dev_dataset(MultiposValidDataset, data_args)
        collator = TrainCollator(data_args, tokenizer)
        trainer = TevatronTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collator,
            dont_shuffle=model_args.dont_shuffle,
            train_group_size=data_args.train_group_size,
            compute_metrics=_compute_ndcg_plain,
        )
    
    else:
        print("***************Using DenseJointLHModel***************")
        model = DenseJointLHModel.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        train_dataset = JointLHDistinctBatchTrainDataset(data_args, 
                                                        effective_batch_size=effective_bs
                                                        )
        dev_dataset = _make_dev_dataset(JointLHTrainDataset, data_args)
        collator = TrainJointLHCollator(data_args, tokenizer)
        trainer = JointLHTevatronTrainer(
            num_positives=data_args.num_positives,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collator,
            dont_shuffle=model_args.dont_shuffle,
            train_group_size=data_args.train_group_size,
            compute_metrics=_compute_ndcg_jointlh,
        )

    train_dataset.trainer = trainer
    if getattr(trainer, "eval_dataset", None) is not None:
        trainer.eval_dataset.trainer = trainer

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
