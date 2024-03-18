import torch
import os
import multiprocessing
from logger import GetLogger
from trainer import CustomDPOTrainer
from typing import Literal, Optional, List, Dict, Any, Union, Sequence, Tuple
from dataclasses import dataclass
from model_loader import load_model_and_tokenizer, create_ref_model
from arguments import DataArguments, FinetuningArguments, ModelArguments
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from preprocess import  get_dpo_dataset
from transformers import DataCollatorForSeq2Seq

IGNORE_INDEX = -100
logger = GetLogger

def get_available_processes():
    try:
        # os.cpu_count()는 논리적인 CPU 코어의 개수를 반환
        return os.cpu_count() or 1
    except NotImplementedError:
        # os.cpu_count()가 지원되지 않는 경우, multiprocessing 모듈을 활용
        return multiprocessing.cpu_count() or 1

@dataclass
class DPODataCollatorWithPadding(DataCollatorForSeq2Seq):
    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        concatenated_features = []
        label_positions = []
        for key in ("chosen_ids", "rejected_ids"):
            for feature in features:
                prompt_len, answer_len = len(feature["prompt_ids"]), len(feature[key])
                concatenated_features.append(
                    {
                        "input_ids": feature["prompt_ids"] + feature[key],
                        "attention_mask": [1] * (prompt_len + answer_len),
                    }
                )
                label_positions.append((prompt_len, answer_len))

        batch = self.tokenizer.pad(
            concatenated_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        return batch

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

    dataset = get_dpo_dataset(tokenizer, model_args, data_args, training_args)

    data_collator = DPODataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Create reference model
    if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
        ref_model = model
    else:
        ref_model = create_ref_model(model_args, finetuning_args)

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(remove_unused_columns=False))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        beta=finetuning_args.dpo_beta,
        loss_type=finetuning_args.dpo_loss,
        ftx_gamma=finetuning_args.dpo_ftx,
        model=model,
        ref_model=ref_model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **{"train_dataset": dataset},
    )
    logger.info('---0---')
    logger.info(dataset[0])
    # Training
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()



if __name__ == '__main__':
    main()