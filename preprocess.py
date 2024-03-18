from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union
from functools import partial
from transformers import Seq2SeqTrainingArguments
from datasets import Dataset, load_dataset
from logger import GetLogger

logger = GetLogger
IGNORE_INDEX = -100

def get_template(user, input='', gpt=''):
    if len(input) >= 1:
        return f"### User:\n{user}\n{input}\n### Assistant:{gpt}\n"
    else:
        return f"### User:\n{user}\n### Assistant:\n"
def infer_max_len(source_len: int, target_len: int, max_len: int, reserved_label_len: int) -> Tuple[int, int]:
    max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - max_target_len
    return max_source_len, max_target_len

def convert_elements_to_ids(
        tokenizer: "PreTrainedTokenizer", elements: List[Union[str, Dict[str, str]]]
) -> List[int]:
    r"""
    Converts elements to token ids.
    """
    token_ids = []
    for elem in elements:
        if isinstance(elem, str):
            if len(elem) != 0:
                token_ids += tokenizer.encode(elem, add_special_tokens=False)
        elif isinstance(elem, dict):
            token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
        elif isinstance(elem, set):
            if "bos_token" in elem and tokenizer.bos_token_id:
                token_ids += [tokenizer.bos_token_id]
            elif "eos_token" in elem and tokenizer.eos_token_id:
                token_ids += [tokenizer.eos_token_id]
        else:
            raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))
    return token_ids

def make_pairs(
        encoded_messages: Sequence[List[int]],
        cutoff_len: int,
        reserved_label_len: int,
) -> Sequence[Tuple[List[int], List[int]]]:
    encoded_pairs = []
    total_length = 0
    for i in range(0, len(encoded_messages), 2):
        if total_length >= cutoff_len:
            break

        max_source_len, max_target_len = infer_max_len(
            source_len=len(encoded_messages[i]),
            target_len=len(encoded_messages[i + 1]),
            max_len=(cutoff_len - total_length),
            reserved_label_len=reserved_label_len,
        )
        encoded_messages[i] = encoded_messages[i][:max_source_len]
        encoded_messages[i + 1] = encoded_messages[i + 1][:max_target_len]
        total_length += len(encoded_messages[i]) + len(encoded_messages[i + 1])
        encoded_pairs.append((encoded_messages[i], encoded_messages[i + 1]))
    return encoded_pairs

def encode_datainfo(
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        cutoff_len=1024,
        reserved_label_len: Optional[int] = 16,
) -> Sequence[Tuple[List[int], List[int]]]:
    user = messages[0]
    gpt = messages[1]
    message = [get_template(user, '', ''), gpt]
    return make_pairs([convert_elements_to_ids(tokenizer, message[0]),
                       convert_elements_to_ids(tokenizer, message[1])], cutoff_len, reserved_label_len)
def preprocess_supervised_dataset(
        examples: Dict[str, List[Any]],
        tokenizer: "PreTrainedTokenizer"
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["prompt"])):
        messages = [examples["prompt"][i]] + [examples["response"][i]]
        input_ids, labels = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(
                encode_datainfo(
                    tokenizer, messages
                )
        ):
            source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs
def convert_sharegpt(examples):
    outputs = {"prompt": [], "response": []}
    for i, messages in enumerate(examples['conversations']):
        messages = messages[: len(messages) // 2 * 2]  # should be multiples of 2
        if len(messages) <= 1:
            continue
        outputs["prompt"].append(messages[0]['value'])
        outputs["response"].append(messages[1]['value'])
    return outputs

def get_sft_dataset(
        tokenizer: "PreTrainedTokenizer",
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments"
) -> Union["Dataset", "IterableDataset"]:
    with training_args.main_process_first(desc="load dataset"):
        kwargs = {"trust_remote_code": True}
        logger.info(f"load_dataset {data_args.dataset}")
        if data_args.dataset.endswith('jsonl'):
            dataset = load_dataset(
                path='json',
                data_files=[data_args.dataset],
                split='train',
                **kwargs,
            )
        else:
            dataset = load_dataset(
                data_args.dataset,
                split='train',
                **kwargs,
            )

        # dataset.cleanup_cache_files()
        convert_func = partial(convert_sharegpt)
        column_names = list(next(iter(dataset)).keys())
        kwargs = dict(
            num_proc=data_args.num_workers,
            desc="load dataset",
        )
        dataset = dataset.map(convert_func, batched=True, remove_columns=column_names, **kwargs)

    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func = partial(
            preprocess_supervised_dataset, tokenizer=tokenizer
        )
        column_names = list(next(iter(dataset)).keys())
        # dataset.cleanup_cache_files()
        dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names)
        return dataset

def preprocess_pdo_dataset(
        examples: Dict[str, List[Any]],
        tokenizer: "PreTrainedTokenizer"
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
    for i in range(len(examples["prompt"])):
        prompt_format = examples["prompt"][i]
        chosen_format = examples["chosen"][i]
        rejected_format = examples["rejected"][i]

        (prompt_ids, chosen_ids) = encode_datainfo(tokenizer, [[prompt_format], [chosen_format]])[0]

        (prompt_ids, rejected_ids) = encode_datainfo(tokenizer, [[prompt_format], [rejected_format]])[0]
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

        model_inputs["prompt_ids"].append(prompt_ids)
        model_inputs["chosen_ids"].append(chosen_ids)
        model_inputs["rejected_ids"].append(rejected_ids)

    return model_inputs


def get_dpo_dataset(tokenizer, model_args, data_args, training_args) -> Union["Dataset", "IterableDataset"]:
    with training_args.main_process_first(desc="load dataset"):
        kwargs = {"trust_remote_code": True}
        logger.info(f"load_dataset {data_args.dataset}")
        if data_args.dataset.endswith('jsonl'):
            dataset = load_dataset(
                path='json',
                data_files=[data_args.dataset],
                split='train',
                **kwargs,
            )
        else:
            dataset = load_dataset(
                data_args.dataset,
                split='train',
                **kwargs,
            )

        def convert_dpo(examples):
            outputs = {"prompt": [], "chosen": [], "rejected": []}

            for i, (instruction, input, output) in enumerate(
                    zip(examples['instruction'], examples['input'], examples['output'])):
                outputs["prompt"].append(get_template(instruction, input, ''))
                outputs["chosen"].append(output[0])
                outputs["rejected"].append(output[1])
            return outputs

        convert_func = partial(convert_dpo)
        column_names = list(next(iter(dataset)).keys())
        kwargs = dict(
            num_proc=data_args.num_workers,
            desc="load dataset",
        )
        dataset = dataset.map(convert_func, batched=True, remove_columns=column_names, **kwargs)

    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func = partial(
            preprocess_pdo_dataset, tokenizer=tokenizer
        )
        column_names = list(next(iter(dataset)).keys())
        # dataset.cleanup_cache_files()
        dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names)
        return dataset
