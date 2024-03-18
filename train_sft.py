import os
import multiprocessing
import torch
import transformers

from logger import GetLogger
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from model_loader import load_model_and_tokenizer
from trainer import CustomSeq2SeqTrainer
from arguments import DataArguments, FinetuningArguments, ModelArguments
from preprocess import  get_sft_dataset

IGNORE_INDEX = -100

logger = GetLogger

def get_available_processes():
    try:
        # os.cpu_count()는 논리적인 CPU 코어의 개수를 반환
        return os.cpu_count() or 1
    except NotImplementedError:
        # os.cpu_count()가 지원되지 않는 경우, multiprocessing 모듈을 활용
        return multiprocessing.cpu_count() or 1


def main():
    logger.info("train_sft")
    parser = HfArgumentParser(
        [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments])
    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    model_args, data_args, training_args, finetuning_args = (*parsed_args,)

    if (training_args.local_rank != -1 and finetuning_args.finetuning_type == "lora"):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args_dict = training_args.to_dict()
        training_args_dict.update(dict(ddp_find_unused_parameters=False))
        training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # postprocess model_args
    model_args.compute_dtype = torch.bfloat16
    model_args.model_max_length = data_args.cutoff_len

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        f"  distributed training: {bool(training_args.local_rank != -1)}, compute dtype: {str(model_args.compute_dtype)}"
        f"  cpu process: {get_available_processes()}")

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)

    dataset = get_sft_dataset(tokenizer, model_args, data_args, training_args)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(
        dict(
            generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
            generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams,
        )
    )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **{"train_dataset": dataset},
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()



if __name__ == '__main__':
    main()