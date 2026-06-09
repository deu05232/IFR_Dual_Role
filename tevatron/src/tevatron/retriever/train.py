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
    
################################ 수정
from dataset import FixedOriginalTrainDataset, FixedOriginalTrainDatasetInstruct, QueryPositionTrainDataset

from collator import TrainCollator

from modeling import DenseModel


from trainer import TevatronTrainer
################################

from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer

logger = logging.getLogger(__name__)


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
        
        
    print("***************Using Dense Model***************")
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )  

    if "-instruct" in training_args.output_dir.lower() and "no_template" not in training_args.output_dir.lower():
        print("*****Instruct 모델 chat template 적용하여 학습*****")
        train_dataset = FixedOriginalTrainDatasetInstruct(data_args, 
                                             tokenizer=tokenizer, 
                                             model_name=model_args.model_name_or_path.lower())
    elif "query_position" in training_args.output_dir.lower():
        print("*****query shuffle하여 학습, 6:2:2*****")
        train_dataset = QueryPositionTrainDataset(data_args)
    else:
        train_dataset = FixedOriginalTrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer)
    trainer_cls = TevatronTrainer
    

    # trainer_cls = GCTrainer if training_args.grad_cache else TevatronTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        dont_shuffle=model_args.dont_shuffle
    )
    train_dataset.trainer = trainer

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
